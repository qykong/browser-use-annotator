"""
Advanced Gradio UI for Browser Use Trajectory Annotation

Initial version borrowed from: https://github.com/trycua/cua/tree/main
"""

import asyncio
import hashlib
import io
import json
import os
from datetime import datetime

import datasets
import pandas as pd
from datasets import Dataset, Features, Sequence
from PIL import Image

import gradio as gr
import gradio.themes as gr_themes
from bua.gradio.constants import (
    LANG,
    LANGUAGES,
    OUTPUT_DIR,
    SESSION_DIR,
    computers,
    generate_session_id,
    get_session_dir,
    initialize_session,
    last_actions,
    last_screenshots,
    screenshot_images,
    tool_call_logs,
)
from bua.gradio.utils import (
    ensure_url_protocol,
    get_chatbot_messages,
    get_last_action_display,
    handle_reasoning_refinement,
    load_all_sessions,
)
from bua.interface.browser import BrowserComputerInterface


def get_sessions_data(session_id):
    """Load all sessions dataset"""

    combined_ds = load_all_sessions(session_id=session_id)
    try:
        if combined_ds:
            # Convert to pandas and select columns
            df = combined_ds.to_pandas()
            columns = ["name", "messages", "source_folder"]
            if "tags" in df.columns:
                columns.append("tags")
            return df[columns]
    except Exception as e:
        print(f"Error loading sessions data: {e}; combined_ds: {combined_ds}")
    
    return pd.DataFrame({"name": [""], "messages": [""], "source_folder": [""]})


def save_demonstration(session_id, log_data, demo_name=None):
    """Save the current tool call logs as a demonstration file using HuggingFace datasets"""
    if session_id not in tool_call_logs or not tool_call_logs[session_id]:
        return "No data to save", None

    # Create output directories if they don't exist
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    if not os.path.exists(SESSION_DIR):
        os.makedirs(SESSION_DIR)

    assert demo_name is not None

    log_time = datetime.now().isoformat()

    # Get session-specific data
    session_tool_logs = tool_call_logs[session_id]
    session_screenshots = screenshot_images.get(session_id, [])

    # Create dataset
    demonstration_dataset = [
        {
            "timestamp": str(log_time),
            "session_id": str(session_id),
            "name": str(demo_name),
            "tool_calls": json.dumps(session_tool_logs),
            "images": [Image.open(io.BytesIO(img)) for img in session_screenshots],
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
        safe_name = (
            demo_name.replace(" ", "_").replace("/", "_").replace("\\", "_")[:50]
        )
        session_folder = os.path.join(
            get_session_dir(session_id), f"{safe_name}_{session_id}_{timestamp}"
        )

        # Create the directory if it doesn't exist
        if not os.path.exists(session_folder):
            os.makedirs(session_folder)

        # Save the dataset to the unique folder
        new_session_ds.save_to_disk(session_folder)

        return f"Session saved to {session_folder}"
    except Exception as e:
        return f"Error saving demonstration: {str(e)}"


def log_tool_call(session_id, name, action, arguments, result=None):
    """Log a tool call with unique IDs and results"""
    if session_id not in tool_call_logs:
        initialize_session(session_id)

    session_tool_logs = tool_call_logs[session_id]
    session_screenshots = screenshot_images[session_id]

    args = {"action": action, **arguments}

    processed_result = {}
    if result:
        for key, value in result.items():
            if key == "screenshot" and isinstance(value, bytes):
                # Add screenshot to the array and get its index
                screenshot_index = len(session_screenshots)
                session_screenshots.append(value)
                # Create hash of screenshot data that includes the index
                hash_value = hashlib.md5(value).hexdigest()
                processed_result[key] = (
                    f"<Screenshot: MD5 {hash_value}:{screenshot_index}>"
                )
            elif key == "clipboard" and isinstance(value, str):
                processed_result[key] = value
            elif isinstance(value, bytes):
                hash_value = hashlib.md5(value).hexdigest()
                processed_result[key] = f"<Binary data: MD5 {hash_value}>"
            else:
                processed_result[key] = value

    log_entry = {
        "type": "function_call",
        "name": name,
        "arguments": json.dumps(args),
        "result": processed_result if result else None,
    }

    session_tool_logs.append(log_entry)
    print(f"Tool call logged: {json.dumps(log_entry)}")

    return log_entry


async def execute(session_id, name, action, arguments):
    """Execute a tool call, log it, and return any results"""
    if session_id not in computers:
        initialize_session(session_id)

    computer = computers[session_id]
    last_screenshot_before = last_screenshots.get(session_id)

    # Store last action for reasoning box
    last_actions[session_id] = {"name": name, "action": action, "arguments": arguments}

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
    log_tool_call(session_id, name, action, arguments, results)

    if "screenshot" in results:
        # Convert bytes to PIL Image
        screenshot_img = Image.open(io.BytesIO(results["screenshot"]))
        results["screenshot"] = screenshot_img
        # Update last_screenshot with the new screenshot
        last_screenshots[session_id] = screenshot_img

    return results


async def handle_init_computer(session_id):
    """Initialize the computer instance and tools for macOS or Ubuntu"""
    if session_id not in computers:
        initialize_session(session_id)

    computers[session_id] = BrowserComputerInterface()
    await computers[session_id].wait_for_ready()

    result = await execute(
        session_id,
        "computer",
        "initialize",
        {},
    )

    return result["screenshot"], json.dumps(tool_call_logs[session_id], indent=2)


async def handle_screenshot(session_id):
    """Take a screenshot and return it as a PIL Image"""
    if session_id not in computers or computers[session_id] is None:
        return None

    result = await execute(session_id, "computer", "screenshot", {})
    return result["screenshot"]


async def handle_wait(session_id):
    """Wait for 1 second and then take a screenshot"""
    if session_id not in computers or computers[session_id] is None:
        return None

    # Execute wait action
    result = await execute(session_id, "computer", "wait", {})
    return result["screenshot"], json.dumps(tool_call_logs[session_id], indent=2)


async def handle_click(session_id, evt: gr.SelectData, img, click_type):
    """Handle click events on the image based on click type"""
    if session_id not in computers or computers[session_id] is None:
        return img, json.dumps(tool_call_logs.get(session_id, []), indent=2)

    # Get the coordinates of the click
    x, y = evt.index

    # Move cursor and perform click
    result = await execute(session_id, "computer", click_type, {"x": x, "y": y})

    # Take a new screenshot to show the result
    return result["screenshot"], json.dumps(tool_call_logs[session_id], indent=2)


async def handle_type(session_id, text, press_enter=False):
    """Type text into the computer"""
    if session_id not in computers or computers[session_id] is None or not text:
        return await handle_screenshot(session_id), json.dumps(
            tool_call_logs.get(session_id, []), indent=2
        )

    result = await execute(
        session_id, "computer", "type_text", {"text": text, "press_enter": press_enter}
    )

    return (
        result["screenshot"],
        json.dumps(tool_call_logs[session_id], indent=2),
        gr.Checkbox(value=False),
    )


async def handle_go_to_url(session_id, url):
    """Go to a URL"""
    if not url:
        return await handle_screenshot(session_id), json.dumps(
            tool_call_logs.get(session_id, []), indent=2
        )

    if session_id not in computers or computers[session_id] is None:
        await handle_init_computer(session_id)
    url = ensure_url_protocol(url)
    result = await execute(session_id, "computer", "go_to_url", {"url": url})

    return result["screenshot"], json.dumps(tool_call_logs[session_id], indent=2)


async def handle_shutdown(session_id):
    """Shutdown the computer instance"""
    if session_id not in computers or computers[session_id] is None:
        return "Computer not initialized", json.dumps(
            tool_call_logs.get(session_id, []), indent=2
        )

    await execute(session_id, "computer", "shutdown", {})

    computers[session_id] = None
    return json.dumps(tool_call_logs[session_id], indent=2), gr.Image(value=None)


async def update_reasoning(session_id, reasoning_text, is_erroneous=False):
    """Update the reasoning for the last action"""
    if session_id not in last_actions or not last_actions[session_id]["name"]:
        return "No action to update reasoning for"

    last_action = last_actions[session_id]
    session_tool_logs = tool_call_logs.get(session_id, [])

    # Find the last log entry that matches the last action
    for log_entry in reversed(session_tool_logs):
        if (
            log_entry["name"] == last_action["name"]
            and json.loads(log_entry["arguments"]).get("action")
            == last_action["action"]
        ):
            # Add reasoning to the log entry
            log_entry["reasoning"] = reasoning_text
            # If marked as erroneous, set weight to 0
            log_entry["weight"] = 0 if is_erroneous else 1
            break

    return "Reasoning updated"


async def clear_log(session_id):
    """Clear the tool call logs"""
    if session_id in screenshot_images:
        screenshot_images[session_id].clear()
    if session_id in tool_call_logs:
        tool_call_logs[session_id].clear()
    return json.dumps(tool_call_logs.get(session_id, []), indent=2)


async def submit_message(session_id, message_text, role, screenshot_after=False):
    """Submit a message with specified role (user or assistant)"""
    # Log the message submission and get result (may include screenshot)
    result = await execute(
        session_id,
        "message",
        "submit",
        {"role": role, "text": message_text, "screenshot_after": screenshot_after},
    )

    # Update return values based on whether a screenshot was taken
    if screenshot_after and "screenshot" in result:
        return (
            f"Message submitted as {role} with screenshot",
            get_chatbot_messages(session_id),
            json.dumps(tool_call_logs[session_id], indent=2),
            result["screenshot"],
        )
    else:
        # Return last screenshot if available
        return (
            f"Message submitted as {role}",
            get_chatbot_messages(session_id),
            json.dumps(tool_call_logs[session_id], indent=2),
            last_screenshots.get(session_id),
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
        # Session management with BrowserState
        session_state = gr.BrowserState(
            "", storage_key="annotation_app", secret="annotation_app"
        )
        gr.Markdown(f"# {LANGUAGES[LANG]['title']}")
        gr.Markdown(
            "🌟 **[Star us on GitHub](https://github.com/qykong/browser-use-annotator)** to support this project!"
        )

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
                        run_setup_btn = gr.Button(
                            LANGUAGES[LANG]["run_task_setup"], variant="primary"
                        )

                with gr.Accordion(
                    LANGUAGES[LANG]["reasoning_last_action"], open=False, visible=False
                ):
                    with gr.Group():
                        last_action_display = gr.Textbox(
                            label=LANGUAGES[LANG]["last_action"],
                            value="",
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
                        reasoning_submit_btn = gr.Button(
                            LANGUAGES[LANG]["submit_reasoning"], variant="primary"
                        )
                        reasoning_refine_btn = gr.Button(
                            LANGUAGES[LANG]["refine"], variant="secondary"
                        )
                    reasoning_status = gr.Textbox(
                        label=LANGUAGES[LANG]["status"], value=""
                    )

                with gr.Accordion(LANGUAGES[LANG]["conversation_messages"], open=False):
                    message_role = gr.Radio(
                        ["user", "assistant"],
                        label=LANGUAGES[LANG]["message_role"],
                        value="user",
                    )
                    message_text = gr.Textbox(
                        label=LANGUAGES[LANG]["message_content"],
                        placeholder=LANGUAGES[LANG]["message_content_placeholder"],
                        lines=3,
                    )
                    screenshot_after_msg = gr.Checkbox(
                        label=LANGUAGES[LANG]["screenshot_after_msg"], value=False
                    )
                    message_submit_btn = gr.Button(
                        LANGUAGES[LANG]["submit_message"], variant="primary"
                    )
                    message_status = gr.Textbox(
                        label=LANGUAGES[LANG]["message_status"], value=""
                    )

                clear_log_btn = gr.Button(
                    LANGUAGES[LANG]["clear_log"], variant="secondary"
                )
                shutdown_btn = gr.Button(
                    LANGUAGES[LANG]["shutdown_computer"], variant="stop"
                )
            with gr.Column(scale=3):
                with gr.Group():
                    with gr.Row(equal_height=True):
                        with gr.Column(scale=3):
                            input_text_url = gr.Textbox(
                                placeholder=LANGUAGES[LANG]["go_to_url"],
                                show_label=False,
                            )
                        with gr.Column(scale=1):
                            submit_url_btn = gr.Button(
                                LANGUAGES[LANG]["submit_url"], variant="primary"
                            )

                    img = gr.Image(
                        type="pil",
                        label=LANGUAGES[LANG]["current_screenshot"],
                        show_label=False,
                        interactive=False,
                    )
                with gr.Group():
                    with gr.Row():
                        wait_btn = gr.Button(
                            LANGUAGES[LANG]["wait"], variant="secondary"
                        )
                        scroll_up_btn = gr.Button(
                            LANGUAGES[LANG]["scroll_up"], variant="secondary"
                        )
                        scroll_down_btn = gr.Button(
                            LANGUAGES[LANG]["scroll_down"], variant="secondary"
                        )
                    with gr.Row(equal_height=True):
                        with gr.Column(scale=3):
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
                                input_text = gr.Textbox(
                                    show_label=False,
                                    placeholder=LANGUAGES[LANG][
                                        "input_text_placeholder"
                                    ],
                                )
                                press_enter_checkbox = gr.Checkbox(
                                    label=LANGUAGES[LANG]["press_enter"], value=False
                                )
                            submit_text_btn = gr.Button(
                                LANGUAGES[LANG]["submit_text"], variant="primary"
                            )

                with gr.Tabs() as logs_tabs:
                    with gr.TabItem(LANGUAGES[LANG]["conversational_logs"]):
                        chat_log = gr.Chatbot(
                            value=get_chatbot_messages,
                            label="Conversation",
                            elem_classes="chatbot",
                            type="messages",
                            height=400,
                            sanitize_html=True,
                        )
                    with gr.TabItem(LANGUAGES[LANG]["function_logs"], visible=False):
                        with gr.Group():
                            action_log = gr.JSON(
                                label=LANGUAGES[LANG]["function_logs"], every=0.2
                            )
                    with gr.TabItem(LANGUAGES[LANG]["save_share_demos"]):
                        with gr.Row():
                            with gr.Column(scale=3):
                                dataset_viewer = gr.DataFrame(
                                    label=LANGUAGES[LANG]["all_sessions"],
                                    interactive=True,
                                )

                            with gr.Column(scale=1):
                                with gr.Group():
                                    demo_name = gr.Textbox(
                                        label=LANGUAGES[LANG]["demo_name"],
                                        value="demo_name_placeholder",
                                        placeholder=LANGUAGES[LANG][
                                            "demo_name_placeholder"
                                        ],
                                    )

                                    save_btn = gr.Button(
                                        LANGUAGES[LANG]["save_current_session"],
                                        variant="primary",
                                    )
                                save_status = gr.Textbox(
                                    label=LANGUAGES[LANG]["save_status"], value=""
                                )

        @app.load(
            inputs=[session_state],
            outputs=[
                last_action_display,
                action_log,
                chat_log,
                session_state,
                dataset_viewer,
            ],
        )
        def initialize_session_on_load(session_id):
            """Initialize session when app loads"""
            if session_id == "":
                session_id = generate_session_id()
            print(f"Initializing session {session_id}")
            initialize_session(session_id)
            return (
                get_last_action_display(session_id),
                json.dumps(tool_call_logs.get(session_id, []), indent=2),
                get_chatbot_messages(session_id),
                session_id,
                get_sessions_data(session_id),
            )

        save_btn.click(
            save_demonstration,
            inputs=[session_state, action_log, demo_name],
            outputs=[save_status],
        )

        # Also update the dataset viewer when saving
        save_btn.click(
            get_sessions_data, inputs=[session_state], outputs=dataset_viewer
        )

        async def run_task_setup(session_id, task_text):
            if session_id not in computers or computers[session_id] is None:
                await handle_init_computer(session_id)

            _, _, logs_json, screenshot = await submit_message(
                session_id, task_text, "user", screenshot_after=True
            )
            gr.Info("Setup complete")
            return screenshot, logs_json

        run_setup_btn.click(
            run_task_setup,
            inputs=[session_state, current_task],
            outputs=[img, action_log],
        )

        action_log.change(
            get_chatbot_messages, inputs=[session_state], outputs=[chat_log]
        )

        img.select(
            handle_click,
            inputs=[session_state, img, click_type],
            outputs=[img, action_log],
        )

        wait_btn.click(handle_wait, inputs=[session_state], outputs=[img, action_log])

        async def handle_scroll(session_id, direction, num_clicks=1):
            """Scroll the page up or down"""
            if session_id not in computers or computers[session_id] is None:
                return None, json.dumps(tool_call_logs.get(session_id, []), indent=2)

            try:
                num_clicks = int(num_clicks)
                if num_clicks < 1:
                    num_clicks = 1
            except (ValueError, TypeError):
                num_clicks = 1

            action = "scroll_up" if direction == "up" else "scroll_down"
            result = await execute(
                session_id, "computer", action, {"clicks": num_clicks}
            )

            return result["screenshot"], json.dumps(
                tool_call_logs.get(session_id, []), indent=2
            )

        scroll_up_btn.click(
            handle_scroll,
            inputs=[session_state, gr.State("up")],
            outputs=[img, action_log],
        )
        scroll_down_btn.click(
            handle_scroll,
            inputs=[session_state, gr.State("down")],
            outputs=[img, action_log],
        )

        submit_text_btn.click(
            handle_type,
            inputs=[session_state, input_text, press_enter_checkbox],
            outputs=[img, action_log, press_enter_checkbox],
        )
        submit_url_btn.click(
            handle_go_to_url,
            inputs=[session_state, input_text_url],
            outputs=[img, action_log],
        )

        async def handle_shutdown_wrapper(session_id):
            return await handle_shutdown(session_id)

        shutdown_btn.click(
            handle_shutdown_wrapper, inputs=[session_state], outputs=[action_log, img]
        )

        async def handle_clear_log_wrapper(session_id):
            return await clear_log(session_id)

        clear_log_btn.click(
            handle_clear_log_wrapper, inputs=[session_state], outputs=action_log
        )
        chat_log.clear(
            handle_clear_log_wrapper, inputs=[session_state], outputs=action_log
        )

        img.select(
            get_last_action_display,
            inputs=[session_state],
            outputs=last_action_display,
        )
        wait_btn.click(
            get_last_action_display,
            inputs=[session_state],
            outputs=last_action_display,
        )
        submit_text_btn.click(
            get_last_action_display,
            inputs=[session_state],
            outputs=last_action_display,
        )
        message_submit_btn.click(
            get_last_action_display,
            inputs=[session_state],
            outputs=last_action_display,
        )
        submit_url_btn.click(
            get_last_action_display,
            inputs=[session_state],
            outputs=last_action_display,
        )
        scroll_down_btn.click(
            get_last_action_display,
            inputs=[session_state],
            outputs=last_action_display,
        )
        scroll_up_btn.click(
            get_last_action_display,
            inputs=[session_state],
            outputs=last_action_display,
        )

        async def handle_reasoning_update(session_id, reasoning, is_erroneous):
            status = await update_reasoning(session_id, reasoning, is_erroneous)
            return status, json.dumps(tool_call_logs.get(session_id, []), indent=2)

        reasoning_submit_btn.click(
            handle_reasoning_update,
            inputs=[session_state, reasoning_text, erroneous_checkbox],
            outputs=[reasoning_status, action_log],
        )

        # Connect the refine buttons to the appropriate handlers
        reasoning_refine_btn.click(
            handle_reasoning_refinement,
            inputs=[reasoning_text, current_task],
            outputs=[reasoning_status, reasoning_text],
        )

        # Handle message submission
        async def handle_message_submit(
            session_id, message_content, role, screenshot_after
        ):
            status, chat_messages, logs_json, screenshot = await submit_message(
                session_id, message_content, role, screenshot_after
            )
            if screenshot:
                return status, chat_messages, logs_json, screenshot
            else:
                return (
                    status,
                    chat_messages,
                    logs_json,
                    last_screenshots.get(session_id),
                )

        message_submit_btn.click(
            handle_message_submit,
            inputs=[session_state, message_text, message_role, screenshot_after_msg],
            outputs=[message_status, chat_log, action_log, img],
        )

    return app


if __name__ == "__main__":
    app = create_gradio_ui()
    app.launch(share=False)
