import gradio as gr
from bua.gradio.constants import LANG, LANGUAGES

def TaskPanel():
    
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
    return {
        "current_task": current_task,
        "run_setup_btn": run_setup_btn,
        "last_action_display": last_action_display,
        "reasoning_text": reasoning_text,
        "erroneous_checkbox": erroneous_checkbox,
        "reasoning_submit_btn": reasoning_submit_btn,
        "reasoning_refine_btn": reasoning_refine_btn,
        "reasoning_status": reasoning_status,
        "message_role": message_role,
        "message_text": message_text,
        "screenshot_after_msg": screenshot_after_msg,
        "message_submit_btn": message_submit_btn,
        "message_status": message_status,
        "clear_log_btn": clear_log_btn,
        "shutdown_btn": shutdown_btn,
    } 