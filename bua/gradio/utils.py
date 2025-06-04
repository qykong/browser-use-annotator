import base64
import glob
import io
import json
import os

from datasets import Dataset, concatenate_datasets
from gradio.components import ChatMessage

from gradio.constants import (
    SESSION_DIR,
    last_action,
    last_screenshot,
    last_screenshot_before,
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

    # Use provided logs if specified, otherwise use global tool_call_logs
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
async def handle_text_refinement(
    text_content, content_type="reasoning", task_text="", use_before=False
):
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
        recent_messages = (
            get_chatbot_messages(tool_call_logs)[-3:]
            if len(get_chatbot_messages(tool_call_logs)) >= 3
            else get_chatbot_messages(tool_call_logs)
        )

        # Format message history with titles when available
        formatted_messages = []
        for msg in recent_messages:
            if msg.metadata and "title" in msg.metadata:
                formatted_messages.append(
                    f"{msg.role} ({msg.metadata['title']}): {msg.content}"
                )
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
<recent_messages>{"\n".join(formatted_messages)}</recent_messages>

Make this into a concise reasoning / self-reflection trace, using "I should/need to/let me/it seems/i see". This trace MUST demonstrate planning extensively before each function call, and reflect extensively on the outcomes of the previous function calls. DO NOT do this entire process by making function calls only, as this can impair your ability to solve the problem and think insightfully.



Provide ONLY the refined explanation text, with no additional commentary or markdown."""

        elif content_type == "memory":
            message_prompt = f"""You are helping refine memory/scratchpad content for an AI assistant.

The screenshot below shows the current state of the computer interface.

TASK: <task_text>{task_text}</task_text>

CURRENT MEMORY CONTENT:
<memory_content>{text_content}</memory_content>

RECENT MESSAGES:
<recent_messages>{"\n".join(formatted_messages)}</recent_messages>

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
<recent_messages>{"\n".join(formatted_messages)}</recent_messages>

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
<recent_messages>{"\n".join(formatted_messages)}</recent_messages>

Improve this text to be more clear, concise, and effective.

Provide ONLY the refined text, with no additional commentary or markdown."""

        # Create messages with the screenshot
        messages = []

        # Add message with image if available
        if screenshot_base64:
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": message_prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{screenshot_base64}"
                            },
                        },
                    ],
                }
            )
        else:
            # Fallback if screenshot isn't available
            messages.append({"role": "user", "content": message_prompt})

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
