import gradio as gr
from bua.gradio.constants import LANG, LANGUAGES
from bua.gradio.utils import get_chatbot_messages

def LogsPanel():
    with gr.Tabs(visible=False) as logs_tabs:
        with gr.TabItem(LANGUAGES[LANG]["conversational_logs"]):
            pass
        with gr.TabItem(LANGUAGES[LANG]["function_logs"], visible=False):
            with gr.Group():
                action_log = gr.JSON(
                    label=LANGUAGES[LANG]["function_logs"], every=0.2
                )
        with gr.TabItem(LANGUAGES[LANG]["save_share_demos"], visible=False):
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
                            placeholder=LANGUAGES[LANG]["demo_name_placeholder"],
                        )
                        save_btn = gr.Button(
                            LANGUAGES[LANG]["save_current_session"],
                            variant="primary",
                        )
                    save_status = gr.Textbox(
                        label=LANGUAGES[LANG]["save_status"], value=""
                    )
    with gr.Group():
        with gr.Row():
            with gr.Column(scale=3):
                chat_log = gr.Chatbot(
                    value=get_chatbot_messages,
                    label="Conversation",
                    elem_classes="chatbot",
                    type="messages",
                    height=400,
                    sanitize_html=True,
                )
            with gr.Column(scale=1):
                with gr.Group():
                    demo_name = gr.Textbox(
                        label=LANGUAGES[LANG]["demo_name"],
                        value="demo_name_placeholder",
                        placeholder=LANGUAGES[LANG]["demo_name_placeholder"],
                    )
                    save_btn = gr.Button(
                        LANGUAGES[LANG]["save_current_session"],
                        variant="primary",
                    )
                save_status = gr.Textbox(
                    label=LANGUAGES[LANG]["save_status"], value=""
                )
                dataset_viewer = gr.DataFrame(
                    label=LANGUAGES[LANG]["all_sessions"],
                    interactive=True,
                )
    return {
        "logs_tabs": logs_tabs,
        "chat_log": chat_log,
        "action_log": action_log,
        "dataset_viewer": dataset_viewer,
        "demo_name": demo_name,
        "save_btn": save_btn,
        "save_status": save_status,
    } 