from gradio_image_annotation import image_annotator

import gradio as gr
from bua.gradio.constants import LANG, LANGUAGES


def MainPanel():
    action_buttons = [
        "left_click",
        "double_click",
        "triple_click",
        "right_click",
        "move_cursor",
        "scroll_up_at_position",
        "scroll_down_at_position",
    ]
    action_gr_btns = []

    with gr.Group():
        with gr.Row(equal_height=True):
            input_text_url = gr.Textbox(
                placeholder=LANGUAGES[LANG]["go_to_url"],
                show_label=False,
                submit_btn=LANGUAGES[LANG]["submit_url"],
            )

        img = image_annotator(
            show_label=False,
            interactive=True,
            show_remove_button=False,
            show_share_button=False,
            show_download_button=False,
            show_clear_button=False,
            single_box=True,
            use_default_label=True,
            box_thickness=1,
            sources=None,
            boxes_alpha=0.5,
            container=False,
            disable_edit_boxes=True,
        )
    with gr.Group():
        with gr.Row():
            wait_btn = gr.Button(LANGUAGES[LANG]["wait"], variant="secondary")
            scroll_up_btn = gr.Button(LANGUAGES[LANG]["scroll_up"], variant="secondary")
            scroll_down_btn = gr.Button(
                LANGUAGES[LANG]["scroll_down"], variant="secondary"
            )
        with gr.Row(equal_height=True):
            with gr.Column(scale=4):
                # with gr.Row():
                    # gr.Markdown("#### Action with BBox", container=True)
                with gr.Row(equal_height=True, variant="compact"):
                    for action_button in action_buttons:
                        btn = gr.DuplicateButton(
                            value=action_button,
                            variant="secondary",
                        )
                        action_gr_btns.append(btn)
            with gr.Column(scale=1):
                with gr.Row():
                    input_text = gr.Textbox(
                        show_label=False,
                        placeholder=LANGUAGES[LANG]["input_text_placeholder"],
                        submit_btn=LANGUAGES[LANG]["submit_text"],
                    )
                    press_enter_checkbox = gr.Checkbox(
                        label=LANGUAGES[LANG]["press_enter"], value=False
                    )

    return {
        "input_text_url": input_text_url,
        "img": img,
        "wait_btn": wait_btn,
        "scroll_up_btn": scroll_up_btn,
        "scroll_down_btn": scroll_down_btn,
        "action_gr_btns": action_gr_btns,
        "input_text": input_text,
        "press_enter_checkbox": press_enter_checkbox,
    }
