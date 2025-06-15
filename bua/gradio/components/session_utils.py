import os
import io
import json
from datetime import datetime
import pandas as pd
from datasets import Dataset, Features, Sequence
from PIL import Image
from bua.gradio.constants import OUTPUT_DIR, SESSION_DIR, get_session_dir, initialize_session, tool_call_logs, screenshot_images
from bua.gradio.utils import load_all_sessions

def get_sessions_data(session_id):
    combined_ds = load_all_sessions(session_id=session_id)
    try:
        if combined_ds:
            df = combined_ds.to_pandas()
            columns = ["messages", "source_folder"]
            return df[columns]
    except Exception as e:
        print(f"Error loading sessions data: {e}; combined_ds: {combined_ds}")
    return pd.DataFrame({"messages": [""], "source_folder": [""]})

def save_demonstration(session_id, log_data, demo_name=None):
    if session_id not in tool_call_logs or not tool_call_logs[session_id]:
        return "No data to save", None
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    if not os.path.exists(SESSION_DIR):
        os.makedirs(SESSION_DIR)
    assert demo_name is not None
    log_time = datetime.now().isoformat()
    session_tool_logs = tool_call_logs[session_id]
    session_screenshots = screenshot_images.get(session_id, [])
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
        new_session_ds = Dataset.from_list(
            demonstration_dataset,
            features=Features(
                {
                    "timestamp": "string",
                    "session_id": "string",
                    "name": "string",
                    "tool_calls": "string",
                    "images": Sequence("image"),
                }
            ),
        )
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = demo_name.replace(" ", "_").replace("/", "_").replace("\\", "_")[:50]
        session_folder = os.path.join(get_session_dir(session_id), f"{safe_name}_{session_id}_{timestamp}")
        if not os.path.exists(session_folder):
            os.makedirs(session_folder)
        new_session_ds.save_to_disk(session_folder)
        return f"Session saved to {session_folder}"
    except Exception as e:
        return f"Error saving demonstration: {str(e)}"

def log_tool_call(session_id, name, action, arguments, result=None, extra_info=None):
    if session_id not in tool_call_logs:
        initialize_session(session_id)
    session_tool_logs = tool_call_logs[session_id]
    session_screenshots = screenshot_images[session_id]
    args = {"action": action, **arguments}
    import hashlib
    processed_result = {}
    if result:
        for key, value in result.items():
            if key == "screenshot" and isinstance(value, bytes):
                screenshot_index = len(session_screenshots)
                session_screenshots.append(value)
                hash_value = hashlib.md5(value).hexdigest()
                processed_result[key] = f"<Screenshot: MD5 {hash_value}:{screenshot_index}>"
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
        "extra_info": extra_info,
    }
    session_tool_logs.append(log_entry)
    print(f"Tool call logged: {json.dumps(log_entry)}")
    return log_entry 