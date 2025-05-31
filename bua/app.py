"""
Advanced Gradio UI for Browser Use Trajectory Annotation

Initial version borrowed from: https://github.com/trycua/cua/tree/main
"""

import gradio as gr
import asyncio
import io
import json
import uuid
import hashlib
import os
import glob
import random
import base64
from datetime import datetime
from PIL import Image
from huggingface_hub import DatasetCard, DatasetCardData
from bua.browser import BrowserInterface
from gradio.components import ChatMessage
import pandas as pd
from datasets import Dataset, Features, Sequence, concatenate_datasets
import datasets
from bua.constants import LANGUAGES, title_mappings


def random_name():
    return f"test-{random.randint(100000, 999999)}"

TASK_EXAMPLES = [
    {
        "task": '',
        "setup": None
    }
]


# Global session ID for tracking this run
session_id = str(uuid.uuid4())

# Global computer instance, tool call logs, memory, and chatbot messages
computer = None
tool_call_logs = []
memory = ""
last_action = {"name": "", "action": "", "arguments": {}}
last_screenshot = None  # Store the most recent screenshot
last_screenshot_before = None  # Store the most [-2]th recent screenshot
screenshot_images = []  # Array to store all screenshot images

# Define a constant for the output directory
OUTPUT_DIR = "examples/output"
SESSION_DIR = os.path.join(OUTPUT_DIR, "sessions")

def load_all_sessions(with_images=False):
    """Load and concatenate all session datasets into a single Dataset"""
    try:
        # Get all session folders
        if not os.path.exists(SESSION_DIR):
            return None
        
        session_folders = glob.glob(os.path.join(SESSION_DIR, "*"))
        if not session_folders:
            return None
        
        # Load each dataset and concatenate
        all_datasets = []
        for folder in session_folders:
            try:
                ds = Dataset.load_from_disk(folder)
                if not with_images:
                    ds = ds.remove_columns('images')
                    
                # Add folder name to identify the source
                folder_name = os.path.basename(folder)
                
                # Process the messages from tool_call_logs
                def process_messages(example):
                    messages_text = []
                    current_role = None
                    
                    # Process the logs if they exist in the example
                    if 'tool_calls' in example:
                        # Use the existing get_chatbot_messages function with explicit logs parameter
                        formatted_msgs = get_chatbot_messages(logs=json.loads(example['tool_calls']))
                        
                        # Process each ChatMessage and extract either title or content
                        for msg in formatted_msgs:
                            # Check if role has changed
                            if msg.role != current_role:
                                # Add a line with the new role if it changed
                                if current_role is not None:  # Skip for the first message
                                    messages_text.append("")  # Add an empty line between role changes
                                messages_text.append(f"{msg.role}")
                                current_role = msg.role
                            
                            # Add the message content
                            if msg.metadata and 'title' in msg.metadata:
                                # Use the title if available
                                messages_text.append(msg.metadata['title'])
                            else:
                                # Use just the content without role prefix since we're adding role headers
                                messages_text.append(msg.content)
                    
                    # Join all messages with newlines
                    all_messages = "\n".join(messages_text)
                    
                    return {
                        **example,
                        "source_folder": folder_name,
                        "messages": all_messages,
                    }
                
                # Apply the processing to each example
                ds = ds.map(process_messages)
                all_datasets.append(ds)
            except Exception as e:
                print(f"Error loading dataset from {folder}: {str(e)}")
        
        if not all_datasets:
            return None
        
        # Concatenate all datasets
        return concatenate_datasets(all_datasets)
    except Exception as e:
        print(f"Error loading sessions: {str(e)}")
        return None

def get_existing_tags():
    """Extract all existing tags from saved demonstrations"""
    all_sessions = load_all_sessions()
    if all_sessions is None:
        return [], []
    
    # Convert to pandas and extract tags
    df = all_sessions.to_pandas()
    
    if 'tags' not in df.columns:
        return []
    
    # Extract all tags and flatten the list
    all_tags = []
    for tags in df['tags'].dropna():
        all_tags += list(tags)
    
    # Remove duplicates and sort
    unique_tags = sorted(list(set(all_tags)))
    return unique_tags, unique_tags

def get_sessions_data():
    """Load all sessions dataset"""

    combined_ds = load_all_sessions()
    if combined_ds:
        # Convert to pandas and select columns
        df = combined_ds.to_pandas()
        columns = ['name', 'messages', 'source_folder']
        if 'tags' in df.columns:
            columns.append('tags')
        return df[columns]
    else:
        return pd.DataFrame({"name": [""], "messages": [""], "source_folder": [""]})

def upload_to_huggingface(dataset_name, visibility, filter_tags=None):
    """Upload sessions to HuggingFace Datasets Hub, optionally filtered by tags
    
    Args:
        dataset_name: Name of the dataset on HuggingFace (format: username/dataset-name)
        visibility: 'public' or 'private'
        filter_tags: List of tags to filter by (optional)
        
    Returns:
        Status message
    """
    try:
        # Check if HF_TOKEN is available
        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token:
            return "Error: HF_TOKEN environment variable not found. Please set it before uploading."
        
        # Check if dataset name is in the correct format
        if not dataset_name or "/" not in dataset_name:
            return "Dataset name must be in the format 'username/dataset-name'"
        
        # Load all sessions
        combined_ds = load_all_sessions(with_images=True)
        if combined_ds is None or len(combined_ds) == 0:
            return "No sessions found to upload."
            
        # If tag filtering is provided, filter the datasets
        if filter_tags:
            # Convert to pandas to filter
            df = combined_ds.to_pandas()
            
            if 'tags' not in df.columns:
                return "No sessions with tags found to filter."
            
            # Get list of source folders for sessions that have any of the selected tags
            matching_folders = []
            for _, row in df.iterrows():
                if not len(row.get('tags')):
                    continue
                if any(tag in list(row.get('tags', [])) for tag in filter_tags):
                    matching_folders.append(row['source_folder'])
            
            if not matching_folders:
                return "No sessions matched the selected tag filters."
            
            # Load only the matching datasets
            filtered_datasets = []
            for folder in matching_folders:
                folder_path = os.path.join(SESSION_DIR, folder)
                if os.path.exists(folder_path):
                    try:
                        ds = Dataset.load_from_disk(folder_path)
                        filtered_datasets.append(ds)
                    except Exception as e:
                        print(f"Error loading dataset from {folder}: {str(e)}")
            
            if not len(filtered_datasets):
                return "Error loading the filtered sessions."
            
            # Create a new combined dataset with just the filtered sessions
            upload_ds = concatenate_datasets(filtered_datasets)
            session_count = len(upload_ds)
        else:
            # Use all sessions
            upload_ds = combined_ds
            session_count = len(upload_ds)
        
        tags = ['cua']
        if isinstance(filter_tags, list):
            tags += filter_tags
        
        # Push to HuggingFace
        upload_ds.push_to_hub(
            dataset_name,
            private=visibility == "private",
            token=hf_token,
            commit_message="(Built with github.com/trycua/cua)"
        )
        
        # Create dataset card
        card_data = DatasetCardData(
            language='en',
            license='mit',
            task_categories=['visual-question-answering'],
            tags=tags
        )
        card = DatasetCard.from_template(
            card_data=card_data,
            template_str="---\n{{ card_data }}\n---\n\n# Uploaded computer interface trajectories\n\nThese trajectories were generated and uploaded using [c/ua](https://github.com/trycua/cua)"
        )
        card.push_to_hub(
            dataset_name,
            commit_message="Cua dataset card"
        )
        
        return f"Successfully uploaded {session_count} sessions to HuggingFace Datasets Hub at https://huggingface.co/datasets/{dataset_name}"
    
    except Exception as e:
        return f"Error uploading to HuggingFace: {str(e)}"

def save_demonstration(log_data, demo_name=None, demo_tags=None):
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
    
    # Process tags
    tags = []
    if demo_tags:
        if isinstance(demo_tags, list):
            tags = demo_tags
        elif isinstance(demo_tags, str):
            # Split by comma if it's a comma-separated string
            tags = [tag.strip() for tag in demo_tags.split(',') if tag.strip()]
    
    log_time = datetime.now().isoformat()
    
    def msg_to_dict(msg: ChatMessage):
        return {
            "role": msg.role,
            "content": str(msg.content),
            "metadata": dict(msg.metadata)
        }
    
    # Create dataset
    demonstration_dataset = [{
        "timestamp": str(log_time),
        "session_id": str(session_id),
        "name": str(demo_name),
        "tool_calls": json.dumps(tool_call_logs),
        "messages": json.dumps([msg_to_dict(msg) for msg in get_chatbot_messages(tool_call_logs)]),
        "tags": list(tags),
        "images": [Image.open(io.BytesIO(img)) for img in screenshot_images],
    }]
    
    try:
        # Create a new HuggingFace dataset from the current session
        new_session_ds = Dataset.from_list(
            demonstration_dataset,
            features=Features({
                'timestamp': datasets.Value('string'),
                'session_id': datasets.Value('string'),
                'name': datasets.Value('string'),
                'tool_calls': datasets.Value('string'),
                'messages': datasets.Value('string'),
                'tags': Sequence(datasets.Value('string')),
                'images': Sequence(datasets.Image()),
            })
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
        "result": processed_result if result else None
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
            await computer.close()
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

    computer = BrowserInterface()

    await computer.wait_for_ready()

    # Log computer initialization as a tool call
    result = await execute("computer", "initialize", {
        "display": "1024x768"
    })

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
    
    return result["screenshot"], json.dumps(tool_call_logs, indent=2)

async def handle_go_to_url(url):
    """Go to a URL"""
    global computer
    if computer is None or not url:
        return await handle_screenshot(), json.dumps(tool_call_logs, indent=2)
    
    result = await execute("computer", "go_to_url", {"url": url})
    
    return result["screenshot"], json.dumps(tool_call_logs, indent=2)


async def handle_copy():
    """Copy selected content to clipboard and return it"""
    global computer
    if computer is None:
        return "Computer not initialized", json.dumps(tool_call_logs, indent=2)
    
    result = await execute("computer", "copy_to_clipboard", {})
    content = result.get("clipboard", "No content copied")
    
    return content, json.dumps(tool_call_logs, indent=2)

async def handle_set_clipboard(text):
    """Set clipboard content"""
    global computer
    if computer is None:
        return "Computer not initialized", json.dumps(tool_call_logs, indent=2)
    
    await execute("computer", "set_clipboard", {"text": text})
    
    return f"Clipboard set to: {text}", json.dumps(tool_call_logs, indent=2)

async def handle_run_command(command):
    """Run a shell command"""
    global computer
    if computer is None:
        return "Computer not initialized", json.dumps(tool_call_logs, indent=2)
    
    # Execute the run_command action and log it
    result = await execute("computer", "run_command", {"command": command})
    
    # Get the result from the computer interface
    stdout, stderr = result.get("stdout"), result.get("stderr")
    
    # Format the output
    output = ""
    if stdout:
        output += f"STDOUT:\n{stdout}\n"
    if stderr:
        output += f"STDERR:\n{stderr}\n"
    
    if not output:
        output = "(No output)"
    
    return output, json.dumps(tool_call_logs, indent=2)

async def handle_shutdown():
    """Shutdown the computer instance"""
    global computer
    if computer is None:
        return "Computer not initialized", json.dumps(tool_call_logs, indent=2)
    
    await execute("computer", "shutdown", {})
    
    computer = None
    return "Computer shut down", json.dumps(tool_call_logs, indent=2)

async def handle_memory(memory_text):
    """Update the global memory"""
    global memory
    await execute("memory", "update", { "memory_text": memory_text })
    memory = memory_text
    return "Memory updated"

async def update_reasoning(reasoning_text, is_erroneous=False):
    """Update the reasoning for the last action"""
    global last_action, tool_call_logs
    
    if not last_action["name"]:
        return "No action to update reasoning for"
    
    # Find the last log entry that matches the last action
    for log_entry in reversed(tool_call_logs):
        if (log_entry["name"] == last_action["name"] and 
            json.loads(log_entry["arguments"]).get("action") == last_action["action"]):
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

def get_last_action_display():
    """Format the last action for display in the reasoning box"""
    global last_action
    if not last_action["name"]:
        return "No actions performed yet"
    
    action_str = f"Tool: {last_action['name']}\nAction: {last_action['action']}"
    
    if last_action["arguments"]:
        args_str = "\nArguments:\n"
        for k, v in last_action["arguments"].items():
            args_str += f"  {k}: {v}\n"
        action_str += args_str
    
    return action_str

def get_memory():
    """Get the current memory"""
    global memory
    return memory

def get_chatbot_messages(logs=None):
    """Format chat messages for gr.Chatbot component
    
    Args:
        logs: Optional list of tool call logs. If None, uses global tool_call_logs.
    
    Returns:
        List of ChatMessage objects
    """
    formatted_messages = []
    
    # Use provided logs if specified, otherwise use global tool_call_logs
    logs_to_process = logs if logs is not None else tool_call_logs
    
    for tool_call in logs_to_process:
        if tool_call['type'] != "function_call":
            continue
        
        name = tool_call['name']
        arguments = json.loads(tool_call['arguments'])
        
        role = tool_call['role'] if 'role' in tool_call else arguments['role'] if 'role' in arguments else 'assistant'
        
        if "reasoning" in tool_call:
            formatted_messages += [ChatMessage(
                role=role,
                content=tool_call['reasoning'],
                metadata={"title": "🧠 Reasoning"}
            )]
        
        # Format tool calls with titles
        if name == "message":
            formatted_messages += [ChatMessage(
                role=role,
                content=arguments['text']
            )]
        else:
            # Format tool calls with a title
            action = arguments.get('action', '')
            
            # Look up title based on name.action or just action
            key = f"{name}.{action}"
            if key in title_mappings:
                title = title_mappings[key]
            elif action in title_mappings:
                title = title_mappings[action]
            else:
                title = f"🛠️ {name.capitalize()}: {action}"
            
            # Always set status to done
            status = "done"
            
            # Format the response content
            content_parts = []
            
            # Add arguments
            if arguments:
                content_parts.append("**Arguments:**")
                for k, v in arguments.items():
                    if k != "action":  # Skip action as it's in the title
                        content_parts.append(f"- {k}: {v}")
            
            # Add results if available
            if tool_call.get('result'):
                content_parts.append("\n**Results:**")
                content_parts.append(f"```json\n{json.dumps(tool_call['result'], indent=4)}\n```")
                # for k, v in tool_call['result'].items():
                #     content_parts.append(f"- {k}: {v}")
            
            # Join all content parts
            content = "\n".join(content_parts)
            
            formatted_messages += [ChatMessage(
                role="assistant",
                content=content,
                metadata={"title": title, "status": status}
            )]
    
    return formatted_messages

async def submit_message(message_text, role, screenshot_after=False):
    """Submit a message with specified role (user or assistant)"""
    global last_screenshot
    
    # Log the message submission and get result (may include screenshot)
    result = await execute("message", "submit", {
        "role": role,
        "text": message_text,
        "screenshot_after": screenshot_after
    })
    
    # Update return values based on whether a screenshot was taken
    if screenshot_after and "screenshot" in result:
        return f"Message submitted as {role} with screenshot", get_chatbot_messages(), json.dumps(tool_call_logs, indent=2), result["screenshot"]
    else:
        # Return last screenshot if available
        return f"Message submitted as {role}", get_chatbot_messages(), json.dumps(tool_call_logs, indent=2), last_screenshot

LANG = "English"

def create_gradio_ui():
    
    def get_label(key):
        return LANGUAGES[LANG][key]
    
    
    with gr.Blocks() as app:

        gr.Markdown( f"# {get_label('title')}")
        

        with gr.Row():
            with gr.Column(scale=3):
                with gr.Group():
                    with gr.Row():
                        start_btn = gr.Button(get_label("initialize_computer"), variant="primary")
                        input_text_url = gr.Textbox(placeholder=get_label("go_to_url"), show_label=False)
                        submit_url_btn = gr.Button(get_label("submit_url"))
                    img = gr.Image(
                        type="pil", 
                        label=get_label("current_screenshot"),
                        show_label=False,
                        interactive=False
                    )
                    
                    # Click type selection
                    click_type = gr.Radio(
                        ["left_click", "double_click", "triple_click", "right_click", "move_cursor", "scroll_up_at_position", "scroll_down_at_position"], 
                        label=get_label("click_type"),
                        value="left_click"
                    )
                    
                    with gr.Row():
                        wait_btn = gr.Button(get_label("wait"))
                        done_btn = gr.Button(get_label("done"))
                        fail_btn = gr.Button(get_label("fail"))
                    
                
                # Tabbed logs: Tool logs, Conversational logs, and Demonstrations
                with gr.Tabs() as logs_tabs:
                    with gr.TabItem(get_label("conversational_logs")):
                        chat_log = gr.Chatbot(
                            value=get_chatbot_messages,
                            label="Conversation",
                            elem_classes="chatbot",
                            height=400,
                            type="messages",
                            sanitize_html=True,
                            allow_tags=True
                        )
                    with gr.TabItem(get_label("function_logs")):
                        with gr.Group():
                            action_log = gr.JSON(
                                label=get_label("function_logs"), 
                                every=0.2
                            )
                            clear_log_btn = gr.Button(get_label("clear_log"))
                    with gr.TabItem(get_label("save_share_demos")):
                        with gr.Row():
                            with gr.Column(scale=3):
                                # Dataset viewer - automatically loads sessions with selection column
                                dataset_viewer = gr.DataFrame(
                                    label=get_label("all_sessions"),
                                    value=get_sessions_data,
                                    show_search='filter',
                                    max_height=300,
                                    interactive=True  # Make it interactive for selection
                                )
                                
                                # HuggingFace Upload UI
                                with gr.Group(visible=True):
                                    gr.Markdown(get_label("upload_sessions_hf"))
                                    with gr.Row():
                                        hf_dataset_name = gr.Textbox(
                                            label=get_label("hf_dataset_name"),
                                            placeholder="username/dataset-name",
                                            info="Format: username/dataset-name"
                                        )
                                        hf_visibility = gr.Radio(
                                            choices=["public", "private"],
                                            label=get_label("dataset_visibility"),
                                            value="private"
                                        )
                                    
                                    # Tag filtering with a single multi-select dropdown
                                    filter_tags = gr.Dropdown(
                                        label=get_label("filter_by_tags"),
                                        choices=get_existing_tags()[0],
                                        multiselect=True,
                                        allow_custom_value=True,
                                        info=get_label("filter_by_tags_info")
                                    )
                                    
                                    # Function to update button text based on selected tags
                                    def get_upload_button_text(selected_tags=None):
                                        if not selected_tags:
                                            # Count all sessions
                                            session_folders = glob.glob(os.path.join(SESSION_DIR, "*"))
                                            count = len(session_folders) if session_folders else 0
                                            return f"Upload {count} Sessions to HuggingFace"
                                        else:
                                            # Count sessions with matching tags
                                            all_sessions = load_all_sessions()
                                            if all_sessions is None:
                                                return "Upload 0 Sessions to HuggingFace"
                                            
                                            df = all_sessions.to_pandas()
                                            if 'tags' not in df.columns:
                                                return "Upload 0 Sessions to HuggingFace"
                                            
                                            # Filter by selected tags (sessions that have ANY of the selected tags)
                                            matching_count = 0
                                            for _, row in df.iterrows():
                                                tags = row.get('tags', [])
                                                if not len(tags):
                                                    continue

                                                # Check if any of the selected tags are in this session's tags
                                                if any(tag in list(row['tags']) for tag in selected_tags):
                                                    matching_count += 1
                                            
                                            return f"Upload {matching_count} Sessions to HuggingFace"
                                    
                                    # Initial button text with all sessions
                                    hf_upload_btn = gr.Button(get_upload_button_text())
                                    
                                    # Update button text when filter changes
                                    def update_button_text(selected_tags):
                                        return get_upload_button_text(selected_tags)
                                    
                                    # Connect filter changes to update button text
                                    filter_tags.change(
                                        update_button_text,
                                        inputs=filter_tags,
                                        outputs=hf_upload_btn
                                    )
                                    
                                    hf_upload_status = gr.Textbox(label=get_label("upload_status"), value="")
                            with gr.Column(scale=1):
                                # Demo name with random name button
                                with gr.Group():
                                    demo_name = gr.Textbox(
                                        label=get_label("demo_name"), 
                                        value="demo_name_placeholder",
                                        placeholder=get_label("demo_name_placeholder")
                                    )
                                    random_name_btn = gr.Button("🎲", scale=1)
                                    
                                    # Demo tags dropdown
                                    demo_tags = gr.Dropdown(
                                        label=get_label("demo_tags"),
                                        choices=get_existing_tags()[0],
                                        multiselect=True,
                                        allow_custom_value=True,
                                        info=get_label("demo_tags_info")
                                    )
                                    
                                    save_btn = gr.Button(get_label("save_current_session"))
                                save_status = gr.Textbox(label=get_label("save_status"), value="")
                                
                                # # Function to update the demo name with a new random name
                                # def update_random_name():
                                #     return generate_random_demo_name()
                                
                                # # Connect random name button
                                # random_name_btn.click(
                                #     update_random_name,
                                #     outputs=[demo_name]
                                # )
                        
            with gr.Column(scale=1):
                with gr.Accordion(get_label("memory_scratchpad"), open=False):
                    with gr.Group():
                        memory_display = gr.Textbox(
                            label=get_label("current_memory"),
                            value=get_memory(),
                            lines=5
                        )
                        with gr.Row():
                            memory_submit_btn = gr.Button(get_label("submit_memory"))
                            memory_refine_btn = gr.Button(get_label("refine"))
                    memory_status = gr.Textbox(label=get_label("status"), value="")
                
                with gr.Accordion(get_label("tasks"), open=True):
                    # Add current task display and controls
                    with gr.Group():
                        current_task = gr.Textbox(
                            label=get_label("current_task"),
                            value=TASK_EXAMPLES[0]["task"],
                            placeholder=get_label("current_task_placeholder"),
                            interactive=True
                        )
                        with gr.Row():
                            randomize_task_btn = gr.Button(get_label("randomize_task"))
                            run_setup_btn = gr.Button(get_label("run_task_setup"))
                    # Setup status textbox
                    setup_status = gr.Textbox(label=get_label("setup_status"), value="")
                
                with gr.Group():
                    input_text = gr.Textbox(label=get_label("type_text"))
                    with gr.Row():
                        press_enter_checkbox = gr.Checkbox(label=get_label("press_enter"), value=False)
                        submit_text_btn = gr.Button(get_label("submit_text"))
                        text_refine_btn = gr.Button(get_label("refine"))


                        
                # with gr.Group():
                #     hotkey_keys = gr.Dropdown(
                #         choices=VALID_KEYS,
                #         label=get_label("select_keys"),
                #         multiselect=True,
                #         show_label=False,
                #         allow_custom_value=True,
                #         info=get_label("select_keys_info")
                #     )
                #     hotkey_btn = gr.Button(get_label("send_hotkeys"))
                
                with gr.Accordion(get_label("scrolling"), open=False):
                    with gr.Group():
                        scroll_clicks = gr.Number(label=get_label("number_of_clicks"), value=1, minimum=1, step=1)
                        with gr.Row():
                            scroll_up_btn = gr.Button(get_label("scroll_up"))
                            scroll_down_btn = gr.Button(get_label("scroll_down"))
                
                with gr.Accordion(get_label("reasoning_last_action"), open=False):
                    with gr.Group():
                        last_action_display = gr.Textbox(
                            label=get_label("last_action"),
                            value=get_last_action_display(),
                            interactive=False
                        )
                        reasoning_text = gr.Textbox(
                            label=get_label("thought_process"),
                            placeholder=get_label("thought_process_placeholder"),
                            lines=3
                        )
                        erroneous_checkbox = gr.Checkbox(
                            label=get_label("mark_erroneous"),
                            value=False
                        )
                        reasoning_submit_btn = gr.Button(get_label("submit_reasoning"))
                        reasoning_refine_btn = gr.Button(get_label("refine"))
                    reasoning_status = gr.Textbox(label=get_label("status"), value="")
                
                with gr.Accordion(get_label("conversation_messages"), open=False):
                    message_role = gr.Radio(
                        ["user", "assistant"],
                        label=get_label("message_role"),
                        value="user"
                    )
                    message_text = gr.Textbox(
                        label=get_label("message_content"),
                        placeholder=get_label("message_content_placeholder"),
                        lines=3
                    )
                    screenshot_after_msg = gr.Checkbox(
                        label=get_label("screenshot_after_msg"), 
                        value=False
                    )
                    message_submit_btn = gr.Button(get_label("submit_message"))
                    message_status = gr.Textbox(label=get_label("message_status"), value="")
                
                shutdown_btn = gr.Button(get_label("shutdown_computer"))

        # Handle save button
        save_btn.click(
            save_demonstration,
            inputs=[action_log, demo_name, demo_tags],
            outputs=[save_status]
        )
        
        # Function to refresh the dataset viewer
        def refresh_dataset_viewer():
            return get_sessions_data()
        
        # Also update the dataset viewer when saving
        save_btn.click(
            refresh_dataset_viewer,
            outputs=dataset_viewer
        )
        
        # Also update the tags dropdown when saving
        save_btn.click(
            get_existing_tags,
            outputs=[demo_tags, filter_tags]
        )
        
        # Handle HuggingFace upload button
        hf_upload_btn.click(
            upload_to_huggingface,
            inputs=[hf_dataset_name, hf_visibility, filter_tags],
            outputs=[hf_upload_status]
        )

        # Function to randomize task
        def randomize_task():
            task_dict = random.choice(TASK_EXAMPLES)
            return task_dict["task"]
        
        # Function to run task setup
        async def run_task_setup(task_text):
            global computer
            
            # Check if computer is initialized
            if computer is None:
                return "Computer not initialized. Please initialize the computer first.", img, action_log
            
            # Find the task dict that matches the current task text
            for task_dict in TASK_EXAMPLES:
                if task_dict["task"] == task_text:
                    try:
                        # Run the setup function with the computer interface and return the result
                        setup_func = task_dict["setup"]
                        if setup_func:
                            await setup_func(computer)
                            
                        # Send initial user message
                        _, _, logs_json, screenshot = await submit_message(
                            task_text, 
                            "user", 
                            screenshot_after=True
                        )
                            
                        return f"Setup complete for: {task_text}", screenshot, logs_json
                    except Exception as e:
                        return f"Error during setup: {str(e)}", img, action_log
            
            _, _, logs_json, screenshot = await submit_message(
                            task_text, 
                            "user", 
                            screenshot_after=True
                        )
            return f"Setup complete for: {task_text}", screenshot, logs_json
        
        # Connect the randomize button to the function
        randomize_task_btn.click(
            randomize_task,
            outputs=[current_task]
        )
        
        # Connect the setup button
        run_setup_btn.click(
            run_task_setup,
            inputs=[current_task],
            outputs=[setup_status, img, action_log]
        )
        
        # Event handlers
        action_log.change(
            get_chatbot_messages,
            outputs=[chat_log]
        )
                
        img.select(handle_click, inputs=[img, click_type], outputs=[img, action_log])
        start_btn.click(handle_init_computer, inputs=[], outputs=[img, action_log])
        wait_btn.click(handle_wait, outputs=[img, action_log])
        
        # # DONE and FAIL buttons just do a placeholder action
        # async def handle_done():
        #     output = await execute("computer", "done", {})
        #     return output["screenshot"], json.dumps(tool_call_logs, indent=2)
        
        # async def handle_fail():
        #     output = await execute("computer", "fail", {})
        #     return output["screenshot"], json.dumps(tool_call_logs, indent=2)
        
        # done_btn.click(handle_done, outputs=[img, action_log])
        # fail_btn.click(handle_fail, outputs=[img, action_log])
        
        # Handle hotkey button
        async def handle_hotkey(selected_keys):
            if not selected_keys or len(selected_keys) == 0:
                return await handle_screenshot(), json.dumps(tool_call_logs, indent=2)
            
            # When multiple keys are selected, the last one is the main key, the rest are modifiers
            if len(selected_keys) > 1:
                key = selected_keys[-1]
                modifiers = selected_keys[:-1]
            else:
                # If only one key is selected, no modifiers
                key = selected_keys[0]
                modifiers = []
            
            output = await execute("computer", "send_hotkey", {"keys": selected_keys})
            return output["screenshot"], json.dumps(tool_call_logs, indent=2)
        
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
        scroll_up_btn.click(
            handle_scroll,
            inputs=[gr.State("up"), scroll_clicks],
            outputs=[img, action_log]
        )
        scroll_down_btn.click(
            handle_scroll,
            inputs=[gr.State("down"), scroll_clicks],
            outputs=[img, action_log]
        )
        
        submit_text_btn.click(handle_type, inputs=[input_text, press_enter_checkbox], outputs=[img, action_log])
        submit_url_btn.click(handle_go_to_url, inputs=[input_text_url], outputs=[img, action_log])
        shutdown_btn.click(handle_shutdown, outputs=[action_log])
        clear_log_btn.click(clear_log, outputs=action_log)
        chat_log.clear(clear_log, outputs=action_log)

        
        # Update last action display after each action
        img.select(lambda *args: get_last_action_display(), outputs=last_action_display)
        start_btn.click(lambda: get_last_action_display(), outputs=last_action_display)
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
            outputs=[reasoning_status, action_log]
        )
        
        # Helper function for text refinement - used for all refine buttons
        async def handle_text_refinement(text_content, content_type="reasoning", task_text="", use_before = False):
            global last_screenshot, last_action, tool_call_logs, last_screenshot_before
            
            screenshot = last_screenshot_before if use_before else last_screenshot
            
            # Check if we have the necessary components
            if not text_content.strip():
                return f"No {content_type} text to refine", text_content
                
            if screenshot is None:
                return "No screenshot available for refinement", text_content
            
            try:
                # Convert the PIL image to base64 if available
                screenshot_base64 = None
                if screenshot:
                    with io.BytesIO() as buffer:
                        screenshot.save(buffer, format="PNG")
                        screenshot_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
                
                # Set up the OpenAI client for refinement
                # Try different API keys from environment in order of preference
                api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OMNI_OPENAI_API_KEY")
                
                if not api_key:
                    return "OpenAI API key not found in environment", text_content
                
                from libs.agent.agent.providers.omni.clients.openai import OpenAIClient
                
                # Create a client - use gpt-4 if available, fall back to 3.5-turbo
                model = "gpt-4.1-2025-04-14"
                
                client = OpenAIClient(
                    api_key=api_key,
                    model=model,
                    max_tokens=1024,
                    temperature=0.2,  # Low temperature for more focused refinement
                )
                
                # Get the last 3 messages from the chat history
                recent_messages = get_chatbot_messages(tool_call_logs)[-3:] if len(get_chatbot_messages(tool_call_logs)) >= 3 else get_chatbot_messages(tool_call_logs)
                
                # Format message history with titles when available
                formatted_messages = []
                for msg in recent_messages:
                    if msg.metadata and 'title' in msg.metadata:
                        formatted_messages.append(f"{msg.role} ({msg.metadata['title']}): {msg.content}")
                    else:
                        formatted_messages.append(f"{msg.role}: {msg.content}")
                
                formatted_messages = [f"<message>{msg}</message>" for msg in formatted_messages]
                
                # Create different prompts based on content type
                if content_type == "reasoning":
                    message_prompt = f"""You are helping refine an explanation about why a specific computer UI action is about to be taken.

The screenshot below shows the state of the screen as I prepare to take this action.

TASK: <task_text>{task_text}</task_text>

ACTION I'M ABOUT TO TAKE:
<action_display>{get_last_action_display()}</action_display>

CURRENT EXPLANATION:
<reasoning_content>{text_content}</reasoning_content>

RECENT MESSAGES:
<recent_messages>{'\n'.join(formatted_messages)}</recent_messages>

Make this into a concise reasoning / self-reflection trace, using "I should/need to/let me/it seems/i see". This trace MUST demonstrate planning extensively before each function call, and reflect extensively on the outcomes of the previous function calls. DO NOT do this entire process by making function calls only, as this can impair your ability to solve the problem and think insightfully.



Provide ONLY the refined explanation text, with no additional commentary or markdown."""
                
                elif content_type == "memory":
                    message_prompt = f"""You are helping refine memory/scratchpad content for an AI assistant.

The screenshot below shows the current state of the computer interface.

TASK: <task_text>{task_text}</task_text>

CURRENT MEMORY CONTENT:
<memory_content>{text_content}</memory_content>

RECENT MESSAGES:
<recent_messages>{'\n'.join(formatted_messages)}</recent_messages>

Refine this memory content to be more clear, organized, and useful for the assistant's task.
- Organize information into logical sections
- Prioritize key facts needed for the task
- Remove unnecessary or redundant information
- Make the format more readable with bullet points or other organizational elements if helpful

Provide ONLY the refined memory text, with no additional commentary or markdown."""
                
                elif content_type == "text":
                    message_prompt = f"""You are helping refine text that will be typed into a computer interface.

The screenshot below shows the current state of the computer interface.

TASK: <task_text>{task_text}</task_text>

CURRENT TEXT TO TYPE:
<text_content>{text_content}</text_content>

RECENT MESSAGES:
<recent_messages>{'\n'.join(formatted_messages)}</recent_messages>

Refine this text to be more effective for the current context:
- Fix any spelling or grammar issues
- Improve clarity and conciseness
- Format appropriately for the context
- Optimize the text for the intended use

Provide ONLY the refined text, with no additional commentary or markdown."""
                
                else:
                    message_prompt = f"""You are helping refine text content.

The screenshot below shows the current state of the computer interface.

CURRENT TEXT:
{text_content}

RECENT MESSAGES:
<recent_messages>{'\n'.join(formatted_messages)}</recent_messages>

Improve this text to be more clear, concise, and effective.

Provide ONLY the refined text, with no additional commentary or markdown."""
                
                # Create messages with the screenshot
                messages = []
                
                # Add message with image if available
                if screenshot_base64:
                    messages.append({
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": message_prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{screenshot_base64}"}
                            }
                        ]
                    })
                else:
                    # Fallback if screenshot isn't available
                    messages.append({
                        "role": "user",
                        "content": message_prompt
                    })
                
                print(message_prompt)
                
                # Make the API call
                response = await client.run_interleaved(
                    messages=messages,
                    system="You are a helpful AI assistant that improves and refines text.",
                )
                
                # Extract the refined text from the response
                if "choices" in response and len(response["choices"]) > 0:
                    refined_text = response["choices"][0]["message"]["content"]
                    return f"{content_type.capitalize()} refined successfully", refined_text
                else:
                    return "Error: Unexpected API response format", text_content
                    
            except Exception as e:
                return f"Error refining {content_type}: {str(e)}", text_content
        
        # Define async wrapper functions for each refine button
        async def handle_reasoning_refinement(reasoning, task):
            return await handle_text_refinement(reasoning, "reasoning", task, use_before=True)

        async def handle_memory_refinement(memory_text, task):
            return await handle_text_refinement(memory_text, "memory", task)

        async def handle_text_input_refinement(text, task):
            return await handle_text_refinement(text, "text", task)

        # Connect the refine buttons to the appropriate handlers
        reasoning_refine_btn.click(
            handle_reasoning_refinement,
            inputs=[reasoning_text, current_task],
            outputs=[reasoning_status, reasoning_text]
        )
        
        # Connect memory refine button
        memory_refine_btn.click(
            handle_memory_refinement,
            inputs=[memory_display, current_task],
            outputs=[memory_status, memory_display]
        )
        
        # Status element for type text section
        with gr.Group():
            type_text_status = gr.Textbox(label="Text Status", value="", visible=False)
            
        # Connect text refine button
        text_refine_btn.click(
            handle_text_input_refinement,
            inputs=[input_text, current_task],
            outputs=[type_text_status, input_text]
        )
        
        # Handle memory submission
        async def handle_memory_update(memory_text):
            status = await handle_memory(memory_text)
            return status, json.dumps(tool_call_logs, indent=2)
            
        memory_submit_btn.click(
            handle_memory_update,
            inputs=memory_display,
            outputs=[memory_status, action_log]
        )
        
        # Handle message submission
        async def handle_message_submit(message_content, role, screenshot_after):
            status, chat_messages, logs_json, screenshot = await submit_message(message_content, role, screenshot_after)
            if screenshot:
                return status, chat_messages, logs_json, screenshot
            else:
                return status, chat_messages, logs_json, last_screenshot
        
        message_submit_btn.click(
            handle_message_submit,
            inputs=[message_text, message_role, screenshot_after_msg], 
            outputs=[message_status, chat_log, action_log, img]
        )

    return app

# Launch the app
if __name__ == "__main__":
    app = create_gradio_ui()
    app.launch()
