import os
import random
import string

from bua.interface.browser import LocalBrowserInterface

LANG = "English"
OUTPUT_DIR = "examples/output"
SESSION_DIR = os.path.join(OUTPUT_DIR, "sessions")

# Session-specific state dictionaries (indexed by session_id)
computers = {}  # session_id -> BrowserComputerInterface | None
tool_call_logs = {}  # session_id -> list
memory = {}  # session_id -> str
last_actions = {}  # session_id -> dict
last_screenshots = {}  # session_id -> image | None
last_screenshots_before = {}  # session_id -> image | None
screenshot_images = {}  # session_id -> list


def generate_session_id():
    """Generate a random session ID"""
    return "".join(random.choices(string.ascii_letters + string.digits, k=16))


def get_session_dir(session_id):
    """Get the session-specific directory"""
    return os.path.join(SESSION_DIR, session_id)


def initialize_session(session_id):
    """Initialize all session-specific state for a new session"""
    if session_id not in computers:
        computers[session_id] = None
        tool_call_logs[session_id] = []
        memory[session_id] = ""
        last_actions[session_id] = {"name": "", "action": "", "arguments": {}}
        last_screenshots[session_id] = None
        last_screenshots_before[session_id] = None
        screenshot_images[session_id] = []

        # Create session-specific directory
        session_dir = get_session_dir(session_id)
        if not os.path.exists(session_dir):
            os.makedirs(session_dir)


def cleanup_session(session_id):
    """Clean up session-specific state when session ends"""
    # Close computer interface if it exists
    if session_id in computers and computers[session_id] is not None:
        computers[session_id].close()

    # Remove from all dictionaries
    computers.pop(session_id, None)
    tool_call_logs.pop(session_id, None)
    memory.pop(session_id, None)
    last_actions.pop(session_id, None)
    last_screenshots.pop(session_id, None)
    last_screenshots_before.pop(session_id, None)
    screenshot_images.pop(session_id, None)


LANGUAGES = {
    "English": {
        "title": "Browser Action Annotation Tool",
        "current_screenshot": "Current Screenshot",
        "click_type": "Click Type",
        "wait": "WAIT",
        "done": "DONE",
        "fail": "FAIL",
        "conversational_logs": "Conversational Logs",
        "function_logs": "Function Logs",
        "clear_log": "Clear Log",
        "save_share_demos": "Save Trajectory",
        "all_sessions": "All Sessions",
        "upload_sessions_hf": "Upload Sessions to HuggingFace",
        "hf_dataset_name": "HuggingFace Dataset Name",
        "dataset_visibility": "Dataset Visibility",
        "filter_by_tags": "Filter by tags (optional)",
        "filter_by_tags_info": "When tags are selected, only demonstrations with those tags will be uploaded. Leave empty to upload all sessions.",
        "demo_name": "Trajectory Name",
        "demo_name_placeholder": "Enter a name for this demonstration",
        "demo_tags": "Demonstration Tags",
        "demo_tags_info": "Select existing tags or create new ones",
        "save_current_session": "Save",
        "save_status": "Save Status",
        "upload_status": "Upload Status",
        "memory_scratchpad": "Memory / Scratchpad",
        "current_memory": "Current Memory",
        "submit_memory": "Submit Memory",
        "refine": "Refine",
        "status": "Status",
        "tasks": "Tasks",
        "current_task": "Current Task",
        "randomize_task": "🎲 Randomize Task",
        "run_task_setup": "⚙️ Run Task Setup",
        "setup_status": "Setup Status",
        "os": "OS",
        "initialize_computer": "Initialize Computer",
        "type_text": "Type Text",
        "press_enter": "Press Enter",
        "submit_text": "Submit Text",
        "go_to_url": "Go to URL",
        "submit_url": "Submit URL",
        "select_keys": "Select Keys",
        "select_keys_info": "Select one or more keys to send as a hotkey",
        "send_hotkeys": "Send Hotkey(s)",
        "scrolling": "Scrolling",
        "number_of_clicks": "Number of Clicks",
        "scroll_up": "Scroll Up",
        "scroll_down": "Scroll Down",
        "reasoning_last_action": "Reasoning for Last Action",
        "last_action": "Last Action",
        "thought_process": "What was your thought process behind this action?",
        "thought_process_placeholder": "Enter your reasoning here...",
        "mark_erroneous": "Mark this action as erroneous (sets weight to 0)",
        "submit_reasoning": "Submit Reasoning",
        "conversation_messages": "Conversation Messages",
        "message_role": "Message Role",
        "message_content": "Message Content",
        "message_content_placeholder": "Enter message here...",
        "screenshot_after_msg": "Receive screenshot after message",
        "submit_message": "Submit Message",
        "message_status": "Message Status",
        "clipboard_operations": "Clipboard Operations",
        "clipboard_content": "Clipboard Content",
        "get_clipboard": "Get Clipboard Content",
        "set_clipboard_text": "Set Clipboard Text",
        "set_clipboard": "Set Clipboard",
        "run_shell_commands": "Run Shell Commands",
        "command_to_run": "Command to run",
        "run_command": "Run Command",
        "command_output": "Command Output",
        "shutdown_computer": "Shutdown Browser",
        "language": "Language",
        "public": "public",
        "private": "private",
        "current_task_placeholder": "Enter current task here...",
        "triple_click": "Triple Click",
        "message_editor": "Message Editor",
        "input_text_placeholder": "Enter text here to type into the browser...",
    },
    "中文": {
        "title": "浏览器操作标注工具",
        "current_screenshot": "当前截图",
        "click_type": "点击类型",
        "wait": "等待",
        "done": "完成",
        "fail": "失败",
        "conversational_logs": "对话日志",
        "function_logs": "功能日志",
        "clear_log": "清除日志",
        "save_share_demos": "保存/分享演示",
        "all_sessions": "所有会话",
        "upload_sessions_hf": "上传会话到HuggingFace",
        "hf_dataset_name": "HuggingFace数据集名称",
        "dataset_visibility": "数据集可见性",
        "filter_by_tags": "按标签筛选（可选）",
        "filter_by_tags_info": "选择标签时，只会上传包含这些标签的演示。留空则上传所有会话。",
        "demo_name": "演示名称",
        "demo_name_placeholder": "输入此演示的名称",
        "demo_tags": "演示标签",
        "demo_tags_info": "选择现有标签或创建新标签",
        "save_current_session": "保存当前会话",
        "save_status": "保存状态",
        "upload_status": "上传状态",
        "memory_scratchpad": "记忆/便签",
        "current_memory": "当前记忆",
        "submit_memory": "提交记忆",
        "refine": "优化",
        "status": "状态",
        "tasks": "任务",
        "current_task": "当前任务",
        "randomize_task": "🎲 随机任务",
        "run_task_setup": "⚙️ 运行任务设置",
        "setup_status": "设置状态",
        "os": "操作系统",
        "initialize_computer": "初始化浏览器",
        "type_text": "输入文字",
        "press_enter": "按回车键",
        "submit_text": "提交文字",
        "go_to_url": "访问网址",
        "submit_url": "提交网址",
        "select_keys": "选择按键",
        "select_keys_info": "选择一个或多个按键作为热键发送",
        "send_hotkeys": "发送热键",
        "scrolling": "滚动",
        "number_of_clicks": "点击次数",
        "scroll_up": "向上滚动",
        "scroll_down": "向下滚动",
        "reasoning_last_action": "最后操作的推理",
        "last_action": "最后操作",
        "thought_process": "您进行此操作的思路是什么？",
        "thought_process_placeholder": "在此输入您的推理...",
        "mark_erroneous": "将此操作标记为错误（权重设为0）",
        "submit_reasoning": "提交推理",
        "conversation_messages": "对话消息",
        "message_role": "消息角色",
        "message_content": "消息内容",
        "message_content_placeholder": "在此输入消息...",
        "screenshot_after_msg": "消息后接收截图",
        "submit_message": "提交消息",
        "message_status": "消息状态",
        "clipboard_operations": "剪贴板操作",
        "clipboard_content": "剪贴板内容",
        "get_clipboard": "获取剪贴板内容",
        "set_clipboard_text": "设置剪贴板文字",
        "set_clipboard": "设置剪贴板",
        "run_shell_commands": "运行Shell命令",
        "command_to_run": "要运行的命令",
        "run_command": "运行命令",
        "command_output": "命令输出",
        "shutdown_computer": "关闭浏览器",
        "language": "语言",
        "public": "公开",
        "private": "私有",
        "current_task_placeholder": "在此输入当前任务...",
        "triple_click": "三击",
        "message_editor": "消息编辑器",
        "input_text_placeholder": "在此输入文字以输入浏览器...",
    },
}

title_mappings = {
    "wait": "⏳ Waiting...",
    "done": "✅ Task Completed",
    "fail": "❌ Task Failed",
    "memory.update": "🧠 Memory Updated",
    "screenshot": "📸 Taking Screenshot",
    "move_cursor": "🖱️ Moving Cursor",
    "left_click": "🖱️ Left Click",
    "right_click": "🖱️ Right Click",
    "double_click": "🖱️ Double Click",
    "type_text": "⌨️ Typing Text",
    "press_key": "⌨️ Pressing Key",
    "send_hotkey": "⌨️ Sending Hotkey",
    "copy_to_clipboard": "📋 Copying to Clipboard",
    "set_clipboard": "📋 Setting Clipboard",
    "run_command": "🖥️ Running Shell Command",
    "initialize": "🚀 Initializing Computer",
    "shutdown": "🛑 Shutting Down",
    "triple_click": "🖱️ Triple Click",
}
