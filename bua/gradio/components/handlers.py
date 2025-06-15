import asyncio
import io
import json

from PIL import Image

import gradio as gr
from bua.gradio.constants import (
    computers,
    initialize_session,
    last_actions,
    last_screenshots,
    screenshot_images,
    tool_call_logs,
)
from bua.gradio.utils import ensure_url_protocol, get_chatbot_messages
from bua.interface.browser import LocalBrowserInterface

from .session_utils import log_tool_call


def format_action_response(result, tool_call_logs):
    if isinstance(result, dict) and "screenshot" in result:
        result = result["screenshot"]
        result = {"image": result, "annotations": []}
    elif isinstance(result, dict) and "image" in result:
        result = result
    else:
        result = {"image": result, "annotations": []}
    return result, json.dumps(tool_call_logs, indent=2)


async def execute(session_id, name, action, arguments, extra_info=None):
    if session_id not in computers:
        initialize_session(session_id)
    computer = computers[session_id]
    results = {}
    if name == "computer":
        if computer is None:
            return {}
        if action == "initialize":
            pass
        elif action == "wait":
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
        elif action == "shutdown":
            computer.close()
        elif action == "go_to_url":
            await computer.go_to_url(arguments["url"])
        elif action == "done" or action == "fail":
            pass
        if action != "shutdown" and "screenshot" not in results:
            results["screenshot"] = await computer.screenshot()
    elif name == "message":
        if action == "submit":
            if arguments.get("screenshot_after", False) and computer is not None:
                results["screenshot"] = await computer.screenshot()
    log_tool_call(session_id, name, action, arguments, results, extra_info)
    if "screenshot" in results:
        screenshot_img = Image.open(io.BytesIO(results["screenshot"]))
        results["screenshot"] = screenshot_img
        last_screenshots[session_id] = screenshot_img
    return results


async def handle_init_computer(session_id):
    if session_id not in computers:
        initialize_session(session_id)
    computers[session_id] = LocalBrowserInterface()
    await computers[session_id].wait_for_ready()
    result = await execute(session_id, "computer", "initialize", {})
    return result["screenshot"], json.dumps(tool_call_logs[session_id], indent=2)


async def handle_screenshot(session_id):
    if session_id not in computers or computers[session_id] is None:
        return None
    result = await execute(session_id, "computer", "screenshot", {})
    return result["screenshot"]


async def handle_wait(session_id):
    if session_id not in computers or computers[session_id] is None:
        return None
    result = await execute(session_id, "computer", "wait", {})
    return format_action_response(result, tool_call_logs[session_id])


async def handle_click(session_id, img, click_type):
    if session_id not in computers or computers[session_id] is None:
        return format_action_response(img, tool_call_logs[session_id])

    if len(img["boxes"]) == 0:
        gr.Info("No box selected")
        return format_action_response(img, tool_call_logs.get(session_id, []))
    assert len(img["boxes"]) == 1, "Only one box is allowed"
    box = img["boxes"][0]
    x1, y1, x2, y2 = box["xmin"], box["ymin"], box["xmax"], box["ymax"]
    center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
    result = await execute(
        session_id,
        "computer",
        click_type,
        {"x": center_x, "y": center_y},
        extra_info={"box": box},
    )
    return format_action_response(result, tool_call_logs[session_id])


async def handle_type(session_id, text, press_enter=False):
    if session_id not in computers or computers[session_id] is None or not text:
        return await handle_screenshot(session_id), json.dumps(
            tool_call_logs.get(session_id, []), indent=2
        )
    result = await execute(
        session_id, "computer", "type_text", {"text": text, "press_enter": press_enter}
    )
    return *format_action_response(result, tool_call_logs[session_id]), gr.Checkbox(
        value=False
    )


async def handle_go_to_url(session_id, url):
    if not url:
        result = await handle_screenshot(session_id)
        return format_action_response(result, tool_call_logs[session_id])
    if session_id not in computers or computers[session_id] is None:
        await handle_init_computer(session_id)
    url = ensure_url_protocol(url)
    result = await execute(session_id, "computer", "go_to_url", {"url": url})

    return format_action_response(result, tool_call_logs[session_id])


async def handle_shutdown(session_id):
    if session_id not in computers or computers[session_id] is None:
        return format_action_response(None, tool_call_logs[session_id])
    await execute(session_id, "computer", "shutdown", {})
    computers[session_id] = None
    return format_action_response(None, tool_call_logs[session_id])


async def update_reasoning(session_id, reasoning_text, is_erroneous=False):
    if session_id not in last_actions or not last_actions[session_id]["name"]:
        return "No action to update reasoning for"
    last_action = last_actions[session_id]
    session_tool_logs = tool_call_logs.get(session_id, [])
    for log_entry in reversed(session_tool_logs):
        if (
            log_entry["name"] == last_action["name"]
            and json.loads(log_entry["arguments"]).get("action")
            == last_action["action"]
        ):
            log_entry["reasoning"] = reasoning_text
            log_entry["weight"] = 0 if is_erroneous else 1
            break
    return "Reasoning updated"


async def clear_log(session_id):
    if session_id in screenshot_images:
        screenshot_images[session_id].clear()
    if session_id in tool_call_logs:
        tool_call_logs[session_id].clear()
    return format_action_response(None, tool_call_logs[session_id])[1]


async def submit_message(session_id, message_text, role, screenshot_after=False):
    result = await execute(
        session_id,
        "message",
        "submit",
        {"role": role, "text": message_text, "screenshot_after": screenshot_after},
    )
    if screenshot_after and "screenshot" in result:
        return (
            f"Message submitted as {role} with screenshot",
            get_chatbot_messages(session_id),
            *format_action_response(result, tool_call_logs[session_id]),
        )
    else:
        return (
            f"Message submitted as {role}",
            get_chatbot_messages(session_id),
            *format_action_response(
                last_screenshots.get(session_id), tool_call_logs[session_id]
            ),
        )
