"""
Advanced Gradio UI for Browser Use Trajectory Annotation

Initial version borrowed from: https://github.com/trycua/cua/tree/main
"""

import asyncio
import hashlib
import io
import json
import os
import uuid
from datetime import datetime

import datasets
import gradio as gr
import pandas as pd
from datasets import Dataset, Features, Sequence
from PIL import Image
import gradio.themes as gr_themes

from bua.gradio.constants import (
    LANG,
    LANGUAGES,
    OUTPUT_DIR,
    SESSION_DIR,
    computer,
    last_action,
    last_screenshot,
    screenshot_images,
    tool_call_logs,
)
from bua.gradio.utils import (
    get_chatbot_messages,
    get_last_action_display,
    handle_reasoning_refinement,
    load_all_sessions,
)
from bua.interface.browser import BrowserComputerInterface

# Global session ID for tracking this run
session_id = str(uuid.uuid4())


def get_sessions_data():
    """Load all sessions dataset"""

    combined_ds = load_all_sessions()
    if combined_ds:
        # Convert to pandas and select columns
        df = combined_ds.to_pandas()
        columns = ["name", "messages", "source_folder"]
        if "tags" in df.columns:
            columns.append("tags")
        return df[columns]
    else:
        return pd.DataFrame({"name": [""], "messages": [""], "source_folder": [""]})


def save_demonstration(log_data, demo_name=None):
    """Save the current tool call logs as a demonstration file using HuggingFace datasets"""
    global tool_call_logs, session_id

    if not tool_call_logs:
        return "No data to save", None

    # Create output directories if they don't exist
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    if not os.path.exists(SESSION_DIR):
        os.makedirs(SESSION_DIR)

    assert demo_name is not None

    log_time = datetime.now().isoformat()

    # Create dataset
    demonstration_dataset = [
        {
            "timestamp": str(log_time),
            "session_id": str(session_id),
            "name": str(demo_name),
            "tool_calls": json.dumps(tool_call_logs),
            "images": [Image.open(io.BytesIO(img)) for img in screenshot_images],
        }
    ]

    try:
        # Create a new HuggingFace dataset from the current session
        new_session_ds = Dataset.from_list(
            demonstration_dataset,
            features=Features(
                {
                    "timestamp": datasets.Value("string"),
                    "session_id": datasets.Value("string"),
                    "name": datasets.Value("string"),
                    "tool_calls": datasets.Value("string"),
                    "images": Sequence(datasets.Image()),
                }
            ),
        )

        # Create a unique folder name with demonstration name, session ID and timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = demo_name.replace(" ", "_").replace("/", "_").replace("\\", "_")[:50]
        session_folder = os.path.join(SESSION_DIR, f"{safe_name}_{session_id}_{timestamp}")

        # Create the directory if it doesn't exist
        if not os.path.exists(session_folder):
            os.makedirs(session_folder)

        # Save the dataset to the unique folder
        new_session_ds.save_to_disk(session_folder)

        return f"Session saved to {session_folder}"
    except Exception as e:
        return f"Error saving demonstration: {str(e)}"


def log_tool_call(name, action, arguments, result=None):
    """Log a tool call with unique IDs and results"""
    global tool_call_logs

    # Create arguments JSON that includes the action
    args = {"action": action, **arguments}

    # Process result for logging
    processed_result = {}
    if result:
        for key, value in result.items():
            if key == "screenshot" and isinstance(value, bytes):
                # Add screenshot to the array and get its index
                screenshot_index = len(screenshot_images)
                screenshot_images.append(value)
                # Create hash of screenshot data that includes the index
                hash_value = hashlib.md5(value).hexdigest()
                processed_result[key] = f"<Screenshot: MD5 {hash_value}:{screenshot_index}>"
            elif key == "clipboard" and isinstance(value, str):
                processed_result[key] = value
            elif isinstance(value, bytes):
                # Create hash for any binary data
                hash_value = hashlib.md5(value).hexdigest()
                processed_result[key] = f"<Binary data: MD5 {hash_value}>"
            else:
                processed_result[key] = value

    # Create the tool call log entry
    log_entry = {
        "type": "function_call",
        "name": name,
        "arguments": json.dumps(args),
        "result": processed_result if result else None,
    }

    # Add to logs and immediately flush by printing
    tool_call_logs.append(log_entry)
    print(f"Tool call logged: {json.dumps(log_entry)}")

    return log_entry


async def execute(name, action, arguments):
    """Execute a tool call, log it, and return any results"""
    global computer, last_action, last_screenshot, last_screenshot_before

    last_screenshot_before = last_screenshot

    # Store last action for reasoning box
    last_action = {"name": name, "action": action, "arguments": arguments}

    results = {}

    # Execute the action based on name and action
    if name == "computer":
        if computer is None:
            return {}

        # Get the method from the computer interface
        if action == "initialize":
            # Already initialized, just log
            pass
        elif action == "wait":
            # Wait for 1 second
            await asyncio.sleep(1)
        elif action == "screenshot":
            pass
        elif action == "move_cursor":
            await computer.move_cursor(arguments["x"], arguments["y"])
            await asyncio.sleep(0.2)
        elif action == "left_click":
            await computer.left_click(arguments["x"], arguments["y"])
            await asyncio.sleep(0.5)
        elif action == "right_click":
            await computer.right_click(arguments["x"], arguments["y"])
            await asyncio.sleep(0.5)
        elif action == "double_click":
            await computer.double_click(arguments["x"], arguments["y"])
            await asyncio.sleep(0.5)
        elif action == "triple_click":
            await computer.triple_click(x=arguments["x"], y=arguments["y"])
            await asyncio.sleep(0.5)
        elif action == "scroll_up_at_position":
            await computer.scroll_up(x=arguments["x"], y=arguments["y"])
            await asyncio.sleep(0.5)
        elif action == "scroll_down_at_position":
            await computer.scroll_down(x=arguments["x"], y=arguments["y"])
            await asyncio.sleep(0.5)
        elif action == "type_text":
            await computer.type_text(arguments["text"])
            await asyncio.sleep(0.3)
            if "press_enter" in arguments and arguments["press_enter"]:
                await computer.press_key("enter")
        elif action == "press_key":
            await computer.press_key(arguments["key"])
            await asyncio.sleep(0.3)
        elif action == "scroll_up":
            await computer.scroll_up(clicks=arguments["clicks"])
            await asyncio.sleep(0.3)
        elif action == "scroll_down":
            await computer.scroll_down(clicks=arguments["clicks"])
            await asyncio.sleep(0.3)
        elif action == "send_hotkey":
            await computer.hotkey(*arguments.get("keys", []))
            await asyncio.sleep(0.3)
        elif action == "copy_to_clipboard":
            results["clipboard"] = await computer.copy_to_clipboard()
        elif action == "set_clipboard":
            await computer.set_clipboard(arguments["text"])
        elif action == "run_command":
            stdout, stderr = await computer.run_command(arguments["command"])
            results["stdout"] = stdout
            results["stderr"] = stderr
        elif action == "shutdown":
            computer.close()
        elif action == "go_to_url":
            await computer.go_to_url(arguments["url"])
        elif action == "done" or action == "fail":
            # Just a marker, doesn't do anything
            pass

        # Add a screenshot to the results for every action (if not already there)
        if action != "shutdown" and "screenshot" not in results:
            results["screenshot"] = await computer.screenshot()
    elif name == "message":
        if action == "submit":
            # No action needed for message submission except logging
            # If requested, take a screenshot after message
            if arguments.get("screenshot_after", False) and computer is not None:
                results["screenshot"] = await computer.screenshot()

    # Log the tool call with results
    log_tool_call(name, action, arguments, results)

    if "screenshot" in results:
        # Convert bytes to PIL Image
        screenshot_img = Image.open(io.BytesIO(results["screenshot"]))
        results["screenshot"] = screenshot_img
        # Update last_screenshot with the new screenshot
        last_screenshot = screenshot_img

    return results


async def handle_init_computer():
    """Initialize the computer instance and tools for macOS or Ubuntu"""
    global computer, tool_call_logs, tools

    computer = BrowserComputerInterface()
    await computer.wait_for_ready()

    result = await execute(
        "computer",
        "initialize",
        {},
    )

    return result["screenshot"], json.dumps(tool_call_logs, indent=2)


async def handle_screenshot():
    """Take a screenshot and return it as a PIL Image"""
    global computer
    if computer is None:
        return None

    result = await execute("computer", "screenshot", {})
    return result["screenshot"]


async def handle_wait():
    """Wait for 1 second and then take a screenshot"""
    global computer
    if computer is None:
        return None

    # Execute wait action
    result = await execute("computer", "wait", {})
    return result["screenshot"], json.dumps(tool_call_logs, indent=2)


async def handle_click(evt: gr.SelectData, img, click_type):
    """Handle click events on the image based on click type"""
    global computer
    if computer is None:
        return img, json.dumps(tool_call_logs, indent=2)

    # Get the coordinates of the click
    x, y = evt.index

    # Move cursor and perform click
    result = await execute("computer", click_type, {"x": x, "y": y})

    # Take a new screenshot to show the result
    return result["screenshot"], json.dumps(tool_call_logs, indent=2)


async def handle_type(text, press_enter=False):
    """Type text into the computer"""
    global computer
    if computer is None or not text:
        return await handle_screenshot(), json.dumps(tool_call_logs, indent=2)

    result = await execute("computer", "type_text", {"text": text, "press_enter": press_enter})

    return result["screenshot"], json.dumps(tool_call_logs, indent=2), gr.Checkbox(value=False)


async def handle_go_to_url(url):
    """Go to a URL"""
    global computer
    if computer is None or not url:
        return await handle_screenshot(), json.dumps(tool_call_logs, indent=2)

    result = await execute("computer", "go_to_url", {"url": url})

    return result["screenshot"], json.dumps(tool_call_logs, indent=2)


async def handle_shutdown():
    """Shutdown the computer instance"""
    global computer
    if computer is None:
        return "Computer not initialized", json.dumps(tool_call_logs, indent=2)

    await execute("computer", "shutdown", {})

    computer = None
    return json.dumps(tool_call_logs, indent=2), gr.Image(value=None)


async def update_reasoning(reasoning_text, is_erroneous=False):
    """Update the reasoning for the last action"""
    global last_action, tool_call_logs

    if not last_action["name"]:
        return "No action to update reasoning for"

    # Find the last log entry that matches the last action
    for log_entry in reversed(tool_call_logs):
        if (
            log_entry["name"] == last_action["name"]
            and json.loads(log_entry["arguments"]).get("action") == last_action["action"]
        ):
            # Add reasoning to the log entry
            log_entry["reasoning"] = reasoning_text
            # If marked as erroneous, set weight to 0
            log_entry["weight"] = 0 if is_erroneous else 1
            break

    return "Reasoning updated"


async def clear_log():
    """Clear the tool call logs"""
    global tool_call_logs, screenshot_images
    screenshot_images = []
    tool_call_logs = []
    return json.dumps(tool_call_logs, indent=2)


async def submit_message(message_text, role, screenshot_after=False):
    """Submit a message with specified role (user or assistant)"""
    global last_screenshot

    # Log the message submission and get result (may include screenshot)
    result = await execute(
        "message",
        "submit",
        {"role": role, "text": message_text, "screenshot_after": screenshot_after},
    )

    # Update return values based on whether a screenshot was taken
    if screenshot_after and "screenshot" in result:
        return (
            f"Message submitted as {role} with screenshot",
            get_chatbot_messages(),
            json.dumps(tool_call_logs, indent=2),
            result["screenshot"],
        )
    else:
        # Return last screenshot if available
        return (
            f"Message submitted as {role}",
            get_chatbot_messages(),
            json.dumps(tool_call_logs, indent=2),
            last_screenshot,
        )


def create_gradio_ui():
    theme = gr_themes.Soft(
        primary_hue=gr_themes.colors.slate,
        secondary_hue=gr_themes.colors.gray,
        neutral_hue=gr_themes.colors.stone,
        text_size=gr_themes.sizes.text_md,
        radius_size=gr_themes.sizes.radius_lg,
    )
    with gr.Blocks(theme=theme) as app:
        gr.Markdown(f"# {LANGUAGES[LANG]['title']}")

        with gr.Row():
            with gr.Column(scale=1):
                with gr.Group():
                    current_task = gr.Textbox(
                        label=LANGUAGES[LANG]["current_task"],
                        value="",
                        placeholder=LANGUAGES[LANG]["current_task_placeholder"],
                        interactive=True,
                    )
                    with gr.Row():
                        run_setup_btn = gr.Button(LANGUAGES[LANG]["run_task_setup"], variant="primary")

                with gr.Accordion(LANGUAGES[LANG]["reasoning_last_action"], open=False, visible=False):
                    with gr.Group():
                        last_action_display = gr.Textbox(
                            label=LANGUAGES[LANG]["last_action"],
                            value=get_last_action_display(),
                            interactive=False,
                        )
                        reasoning_text = gr.Textbox(
                            label=LANGUAGES[LANG]["thought_process"],
                            placeholder=LANGUAGES[LANG]["thought_process_placeholder"],
                            lines=3,
                        )
                        erroneous_checkbox = gr.Checkbox(
                            label=LANGUAGES[LANG]["mark_erroneous"], value=False
                        )
                        reasoning_submit_btn = gr.Button(LANGUAGES[LANG]["submit_reasoning"], variant="primary")
                        reasoning_refine_btn = gr.Button(LANGUAGES[LANG]["refine"], variant="secondary")
                    reasoning_status = gr.Textbox(label=LANGUAGES[LANG]["status"], value="")

                with gr.Accordion(LANGUAGES[LANG]["conversation_messages"], open=False):
                    message_role = gr.Radio(
                        ["user", "assistant"], label=LANGUAGES[LANG]["message_role"], value="user"
                    )
                    message_text = gr.Textbox(
                        label=LANGUAGES[LANG]["message_content"],
                        placeholder=LANGUAGES[LANG]["message_content_placeholder"],
                        lines=3,
                    )
                    screenshot_after_msg = gr.Checkbox(
                        label=LANGUAGES[LANG]["screenshot_after_msg"], value=False
                    )
                    message_submit_btn = gr.Button(LANGUAGES[LANG]["submit_message"], variant="primary")
                    message_status = gr.Textbox(label=LANGUAGES[LANG]["message_status"], value="")

                shutdown_btn = gr.Button(LANGUAGES[LANG]["shutdown_computer"], variant="stop")
            with gr.Column(scale=3):
                with gr.Group():
                    with gr.Row(equal_height=True):
                        with gr.Column(scale=3):
                            input_text_url = gr.Textbox(
                                placeholder=LANGUAGES[LANG]["go_to_url"], show_label=False
                            )
                        with gr.Column(scale=1):
                            submit_url_btn = gr.Button(LANGUAGES[LANG]["submit_url"], variant="primary")

                    img = gr.Image(
                        type="pil",
                        label=LANGUAGES[LANG]["current_screenshot"],
                        show_label=False,
                        interactive=False,
                    )
                    with gr.Row():
                        wait_btn = gr.Button(LANGUAGES[LANG]["wait"], variant="secondary")
                        scroll_up_btn = gr.Button(LANGUAGES[LANG]["scroll_up"], variant="secondary")
                        scroll_down_btn = gr.Button(LANGUAGES[LANG]["scroll_down"], variant="secondary")
                    with gr.Row(equal_height=True):
                        with gr.Column(scale=3):
                            # Click type selection
                            click_type = gr.Radio(
                                [
                                    "left_click",
                                    "double_click",
                                    "triple_click",
                                    "right_click",
                                    "move_cursor",
                                    "scroll_up_at_position",
                                    "scroll_down_at_position",
                                ],
                                label=LANGUAGES[LANG]["click_type"],
                                value="left_click",
                            )
                        with gr.Column(scale=1):
                            with gr.Row():
                                input_text = gr.Textbox(show_label=False, placeholder=LANGUAGES[LANG]["input_text_placeholder"])
                                press_enter_checkbox = gr.Checkbox(
                                    label=LANGUAGES[LANG]["press_enter"], value=False
                                )
                            submit_text_btn = gr.Button(LANGUAGES[LANG]["submit_text"], variant="primary")

                # Tabbed logs: Tool logs, Conversational logs, and Demonstrations
                with gr.Tabs() as logs_tabs:
                    with gr.TabItem(LANGUAGES[LANG]["conversational_logs"]):
                        chat_log = gr.Chatbot(
                            value=get_chatbot_messages,
                            label="Conversation",
                            elem_classes="chatbot",
                            height=400,
                            sanitize_html=True,
                        )
                    with gr.TabItem(LANGUAGES[LANG]["function_logs"], visible=False):
                        with gr.Group():
                            action_log = gr.JSON(label=LANGUAGES[LANG]["function_logs"], every=0.2)
                            clear_log_btn = gr.Button(LANGUAGES[LANG]["clear_log"], variant="secondary")
                    with gr.TabItem(LANGUAGES[LANG]["save_share_demos"]):
                        with gr.Row():
                            with gr.Column(scale=3):
                                # Dataset viewer - automatically loads sessions with selection column
                                dataset_viewer = gr.DataFrame(
                                    label=LANGUAGES[LANG]["all_sessions"],
                                    value=get_sessions_data,
                                    interactive=True,
                                )

                            with gr.Column(scale=1):
                                # Demo name with random name button
                                with gr.Group():
                                    demo_name = gr.Textbox(
                                        label=LANGUAGES[LANG]["demo_name"],
                                        value="demo_name_placeholder",
                                        placeholder=LANGUAGES[LANG]["demo_name_placeholder"],
                                    )

                                    save_btn = gr.Button(LANGUAGES[LANG]["save_current_session"], variant="primary")
                                save_status = gr.Textbox(label=LANGUAGES[LANG]["save_status"], value="")

        # Handle save button
        save_btn.click(save_demonstration, inputs=[action_log, demo_name], outputs=[save_status])

        # Function to refresh the dataset viewer
        def refresh_dataset_viewer():
            return get_sessions_data()

        # Also update the dataset viewer when saving
        save_btn.click(refresh_dataset_viewer, outputs=dataset_viewer)

        # Function to run task setup
        async def run_task_setup(task_text):
            global computer

            await handle_init_computer()

            _, _, logs_json, screenshot = await submit_message(
                task_text, "user", screenshot_after=True
            )
            gr.Info("Setup complete")
            return screenshot, logs_json

        # Connect the setup button
        run_setup_btn.click(run_task_setup, inputs=[current_task], outputs=[img, action_log])

        # Event handlers
        action_log.change(get_chatbot_messages, outputs=[chat_log])

        img.select(handle_click, inputs=[img, click_type], outputs=[img, action_log])
        wait_btn.click(handle_wait, outputs=[img, action_log])

        # Define async handler for scrolling
        async def handle_scroll(direction, num_clicks=1):
            """Scroll the page up or down"""
            global computer
            if computer is None:
                return None, json.dumps(tool_call_logs, indent=2)

            # Convert num_clicks to integer with validation
            try:
                num_clicks = int(num_clicks)
                if num_clicks < 1:
                    num_clicks = 1
            except (ValueError, TypeError):
                num_clicks = 1

            # Execute the scroll action
            action = "scroll_up" if direction == "up" else "scroll_down"
            result = await execute("computer", action, {"clicks": num_clicks})

            return result["screenshot"], json.dumps(tool_call_logs, indent=2)

        # Connect scroll buttons
        scroll_up_btn.click(handle_scroll, inputs=[gr.State("up")], outputs=[img, action_log])
        scroll_down_btn.click(handle_scroll, inputs=[gr.State("down")], outputs=[img, action_log])

        submit_text_btn.click(
            handle_type,
            inputs=[input_text, press_enter_checkbox],
            outputs=[img, action_log, press_enter_checkbox],
        )
        submit_url_btn.click(handle_go_to_url, inputs=[input_text_url], outputs=[img, action_log])
        shutdown_btn.click(handle_shutdown, outputs=[action_log, img])
        clear_log_btn.click(clear_log, outputs=action_log)
        chat_log.clear(clear_log, outputs=action_log)

        # Update last action display after each action
        img.select(lambda *args: get_last_action_display(), outputs=last_action_display)
        wait_btn.click(lambda: get_last_action_display(), outputs=last_action_display)
        submit_text_btn.click(lambda: get_last_action_display(), outputs=last_action_display)
        message_submit_btn.click(lambda: get_last_action_display(), outputs=last_action_display)
        submit_url_btn.click(lambda: get_last_action_display(), outputs=last_action_display)
        scroll_down_btn.click(lambda: get_last_action_display(), outputs=last_action_display)
        scroll_up_btn.click(lambda: get_last_action_display(), outputs=last_action_display)

        # Handle reasoning submission
        async def handle_reasoning_update(reasoning, is_erroneous):
            status = await update_reasoning(reasoning, is_erroneous)
            return status, json.dumps(tool_call_logs, indent=2)

        reasoning_submit_btn.click(
            handle_reasoning_update,
            inputs=[reasoning_text, erroneous_checkbox],
            outputs=[reasoning_status, action_log],
        )

        # Connect the refine buttons to the appropriate handlers
        reasoning_refine_btn.click(
            handle_reasoning_refinement,
            inputs=[reasoning_text, current_task],
            outputs=[reasoning_status, reasoning_text],
        )

        # Handle message submission
        async def handle_message_submit(message_content, role, screenshot_after):
            status, chat_messages, logs_json, screenshot = await submit_message(
                message_content, role, screenshot_after
            )
            if screenshot:
                return status, chat_messages, logs_json, screenshot
            else:
                return status, chat_messages, logs_json, last_screenshot

        message_submit_btn.click(
            handle_message_submit,
            inputs=[message_text, message_role, screenshot_after_msg],
            outputs=[message_status, chat_log, action_log, img],
        )

    return app


# Launch the app
if __name__ == "__main__":
    app = create_gradio_ui()
    app.launch(share=False)
