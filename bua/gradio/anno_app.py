"""
Advanced Gradio UI for Browser Use Trajectory Annotation

Initial version borrowed from: https://github.com/trycua/cua/tree/main
"""

import gradio as gr
import gradio.themes as gr_themes
from bua.gradio.constants import (
    LANG,
    LANGUAGES,
    generate_session_id,
    tool_call_logs,
    initialize_session,
)
from bua.gradio.components.task_panel import TaskPanel
from bua.gradio.components.main_panel import MainPanel
from bua.gradio.components.logs_panel import LogsPanel
from bua.gradio.components.session_utils import get_sessions_data, save_demonstration
from bua.gradio.components.handlers import (
    handle_init_computer,
    handle_screenshot,
    handle_wait,
    handle_click,
    handle_type,
    handle_go_to_url,
    handle_shutdown,
    update_reasoning,
    clear_log,
    submit_message,
)
from bua.gradio.utils import (
    get_last_action_display,
    handle_reasoning_refinement,
    get_chatbot_messages,
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
        session_state = gr.BrowserState(
            "", storage_key="annotation_app", secret="annotation_app"
        )
        gr.Markdown(f"# {LANGUAGES[LANG]['title']}")
        gr.Markdown(
            "🌟 **[Star us on GitHub](https://github.com/qykong/browser-use-annotator)** to support this project!"
        )
        with gr.Row():
            with gr.Column(scale=1):
                task = TaskPanel()
            with gr.Column(scale=3):
                main = MainPanel()
                logs = LogsPanel()

        @app.load(
            inputs=[session_state],
            outputs=[
                task["last_action_display"],
                logs["action_log"],
                logs["chat_log"],
                session_state,
                logs["dataset_viewer"],
            ],
        )
        def initialize_session_on_load(session_id):
            if session_id == "":
                session_id = generate_session_id()
            print(f"Initializing session {session_id}")
            initialize_session(session_id)
            return (
                get_last_action_display(session_id),
                tool_call_logs.get(session_id, []),
                logs["chat_log"].value,
                session_id,
                get_sessions_data(session_id),
            )

        for btn in main["action_gr_btns"]:
            btn.click(
                handle_click,
                inputs=[session_state, main["img"], btn],
                outputs=[main["img"], logs["action_log"]],
            )
            btn.click(
                get_last_action_display,
                inputs=[session_state],
                outputs=[task["last_action_display"]],
            )
        logs["save_btn"].click(
            save_demonstration,
            inputs=[session_state, logs["action_log"], logs["demo_name"]],
            outputs=[logs["save_status"]],
        )
        logs["save_btn"].click(
            get_sessions_data, inputs=[session_state], outputs=logs["dataset_viewer"]
        )

        async def run_task_setup(session_id, task_text):
            await handle_init_computer(session_id)
            _, _, screenshot, logs_json = await submit_message(
                session_id, task_text, "user", screenshot_after=True
            )
            gr.Info("Setup complete")
            return screenshot, logs_json

        task["run_setup_btn"].click(
            run_task_setup,
            inputs=[session_state, task["current_task"]],
            outputs=[main["img"], logs["action_log"]],
        )
        logs["action_log"].change(
            get_chatbot_messages,
            inputs=[session_state],
            outputs=[logs["chat_log"]],
        )
        main["wait_btn"].click(
            handle_wait,
            inputs=[session_state],
            outputs=[main["img"], logs["action_log"]],
        )
        main["scroll_up_btn"].click(
            lambda session_id: handle_wait(session_id),
            inputs=[session_state],
            outputs=[main["img"], logs["action_log"]],
        )
        main["scroll_down_btn"].click(
            lambda session_id: handle_wait(session_id),
            inputs=[session_state],
            outputs=[main["img"], logs["action_log"]],
        )
        main["input_text"].submit(
            handle_type,
            inputs=[session_state, main["input_text"], main["press_enter_checkbox"]],
            outputs=[main["img"], logs["action_log"], main["press_enter_checkbox"]],
        )
        main["input_text_url"].submit(
            handle_go_to_url,
            inputs=[session_state, main["input_text_url"]],
            outputs=[main["img"], logs["action_log"]],
        )
        task["shutdown_btn"].click(
            handle_shutdown,
            inputs=[session_state],
            outputs=[main["img"], logs["action_log"]],
        )
        task["clear_log_btn"].click(
            clear_log, inputs=[session_state], outputs=[logs["action_log"]]
        )
        logs["chat_log"].clear(
            clear_log, inputs=[session_state], outputs=[logs["action_log"]]
        )
        main["wait_btn"].click(
            get_last_action_display,
            inputs=[session_state],
            outputs=[task["last_action_display"]],
        )
        main["input_text"].submit(
            get_last_action_display,
            inputs=[session_state],
            outputs=[task["last_action_display"]],
        )
        task["message_submit_btn"].click(
            get_last_action_display,
            inputs=[session_state],
            outputs=[task["last_action_display"]],
        )
        main["input_text_url"].submit(
            get_last_action_display,
            inputs=[session_state],
            outputs=[task["last_action_display"]],
        )
        main["scroll_down_btn"].click(
            get_last_action_display,
            inputs=[session_state],
            outputs=[task["last_action_display"]],
        )
        main["scroll_up_btn"].click(
            get_last_action_display,
            inputs=[session_state],
            outputs=[task["last_action_display"]],
        )

        async def handle_reasoning_update(session_id, reasoning, is_erroneous):
            status = await update_reasoning(session_id, reasoning, is_erroneous)
            return status, tool_call_logs.get(session_id, [])

        task["reasoning_submit_btn"].click(
            handle_reasoning_update,
            inputs=[session_state, task["reasoning_text"], task["erroneous_checkbox"]],
            outputs=[task["reasoning_status"], logs["action_log"]],
        )
        task["reasoning_refine_btn"].click(
            handle_reasoning_refinement,
            inputs=[task["reasoning_text"], task["current_task"]],
            outputs=[task["reasoning_status"], task["reasoning_text"]],
        )

        async def handle_message_submit(
            session_id, message_content, role, screenshot_after
        ):
            (
                status,
                chat_messages,
                screenshot,
                logs_json,
            ) = await submit_message(
                session_id, message_content, role, screenshot_after
            )
            return status, chat_messages, screenshot, logs_json

        task["message_submit_btn"].click(
            handle_message_submit,
            inputs=[
                session_state,
                task["message_text"],
                task["message_role"],
                task["screenshot_after_msg"],
            ],
            outputs=[
                task["message_status"],
                logs["chat_log"],
                main["img"],
                logs["action_log"],
            ],
        )
    return app


if __name__ == "__main__":
    app = create_gradio_ui()
    app.launch(share=False)
