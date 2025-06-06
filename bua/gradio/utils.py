import base64
import glob
import io
import json
import os

from datasets import Dataset, concatenate_datasets
from gradio.components import ChatMessage

from bua.gradio.constants import (
    SESSION_DIR,
    last_action,
    title_mappings,
    tool_call_logs,
)


def get_chatbot_messages(logs=None):
    """Format chat messages for gr.Chatbot component

    Args:
        logs: Optional list of tool call logs. If None, uses global tool_call_logs.

    Returns:
        List of ChatMessage objects
    """
    formatted_messages = []

    logs_to_process = logs if logs is not None else tool_call_logs

    for tool_call in logs_to_process:
        if tool_call["type"] != "function_call":
            continue

        name = tool_call["name"]
        arguments = json.loads(tool_call["arguments"])

        role = (
            tool_call["role"]
            if "role" in tool_call
            else arguments["role"]
            if "role" in arguments
            else "assistant"
        )

        if "reasoning" in tool_call:
            formatted_messages += [
                ChatMessage(
                    role=role, content=tool_call["reasoning"], metadata={"title": "🧠 Reasoning"}
                )
            ]

        # Format tool calls with titles
        if name == "message":
            formatted_messages += [ChatMessage(role=role, content=arguments["text"])]
        else:
            # Format tool calls with a title
            action = arguments.get("action", "")

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
            if tool_call.get("result"):
                content_parts.append("\n**Results:**")
                content_parts.append(f"```json\n{json.dumps(tool_call['result'], indent=4)}\n```")
                # for k, v in tool_call['result'].items():
                #     content_parts.append(f"- {k}: {v}")

            # Join all content parts
            content = "\n".join(content_parts)

            formatted_messages += [
                ChatMessage(
                    role="assistant", content=content, metadata={"title": title, "status": status}
                )
            ]

    return formatted_messages


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
                    ds = ds.remove_columns("images")

                # Add folder name to identify the source
                folder_name = os.path.basename(folder)

                # Process the messages from tool_call_logs
                def process_messages(example):
                    messages_text = []
                    current_role = None

                    # Process the logs if they exist in the example
                    if "tool_calls" in example:
                        # Use the existing get_chatbot_messages function with explicit logs parameter
                        formatted_msgs = get_chatbot_messages(
                            logs=json.loads(example["tool_calls"])
                        )

                        # Process each ChatMessage and extract either title or content
                        for msg in formatted_msgs:
                            # Check if role has changed
                            if msg.role != current_role:
                                # Add a line with the new role if it changed
                                if current_role is not None:  # Skip for the first message
                                    messages_text.append(
                                        ""
                                    )  # Add an empty line between role changes
                                messages_text.append(f"{msg.role}")
                                current_role = msg.role

                            # Add the message content
                            if msg.metadata and "title" in msg.metadata:
                                # Use the title if available
                                messages_text.append(msg.metadata["title"])
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

# Helper function for text refinement - used for all refine buttons
async def handle_text_refinement():
    raise NotImplementedError


# Define async wrapper functions for each refine button
async def handle_reasoning_refinement(reasoning, task):
    return await handle_text_refinement(reasoning, "reasoning", task, use_before=True)


def ensure_url_protocol(raw_url, default_protocol = "https"):
    """Ensure the url has a protocol"""
    if not raw_url:
        return raw_url

    # Strip whitespace
    url = raw_url.strip()

    # Check if URL already has a protocol
    if url.startswith(("http://", "https://", "ftp://")):
        return url

    # Add default protocol if none exists
    return f"{default_protocol}://{url}"
