import json
import os

import datasets
import gradio as gr
from PIL import Image, ImageDraw, ImageFont
import gradio.themes as gr_themes

from bua.gradio.constants import SESSION_DIR
from bua.gradio.utils import load_all_sessions

tool_calls_list: list[list[dict]] | None = None
images_list: list[list[Image.Image]] | None = None
dataset: datasets.Dataset | datasets.DatasetDict | None = None
current_image_index = 2
current_tool_call_index = 2
prompt_text = ""
tool_calls_index = 0
prompt_text_idx = 0


def plot_dot_to_image(prev_image, result_image, action):
    if "x" in action and "y" in action:
        assert prev_image is not None
        prev_image = prev_image.convert("RGBA")
        draw = ImageDraw.Draw(prev_image)

        # Draw circular indicator (similar to the JS version)
        x, y = action["x"], action["y"]
        radius = 5

        # Draw filled circle with semi-transparent red background
        draw.ellipse(
            [x - radius, y - radius, x + radius, y + radius],
            fill=(0, 0, 0, 128),  # rgba(255, 0, 0, 0.5) equivalent
            outline="black",
            width=2,
        )

        # Add coordinate label
        label_text = f"({x}, {y})"
        label_x, label_y = x + 15, y - 10

        # Draw background for text label
        bbox = draw.textbbox((label_x, label_y), label_text)
        draw.rectangle(
            [bbox[0] - 2, bbox[1] - 2, bbox[2] + 2, bbox[3] + 2],
            fill="black",
        )

        # Draw the coordinate text
        draw.text((label_x, label_y), label_text, fill="white")

    if prev_image is None:
        prev_image = Image.new("RGBA", result_image.size, (255, 255, 255, 255))
    else:
        prev_image = prev_image.convert("RGBA")
    result_image = result_image.convert("RGBA")

    width, height = result_image.size
    label_height = 30
    border_width = 3
    combined = Image.new(
        "RGBA", (width * 2, height + label_height), (255, 255, 255, 255)
    )

    # Draw labels
    draw_combined = ImageDraw.Draw(combined)

    draw_combined.text((width // 2 - 60, 5), "Before Action", fill="black", font_size=20)
    draw_combined.text(
        (width + width // 2 - 60, 5), "After Action", fill="black", font_size=20
    )

    # Paste images
    combined.paste(prev_image, (0, label_height))
    combined.paste(result_image, (width, label_height))

    # Draw borders around both images
    # Border for before image
    draw_combined.rectangle(
        [0, label_height, width - 1, label_height + height - 1],
        outline="black", width=border_width
    )
    # Border for after image
    draw_combined.rectangle(
        [width, label_height, width * 2 - 1, label_height + height - 1],
        outline="black", width=border_width
    )

    return combined


def plot_image(prev_image, image, tool_calls, image_index):
    global current_tool_call_index
    action = None
    reasoning = ""
    for i, tool_call in enumerate(tool_calls):
        if (
            "result" in tool_call
            and tool_call["result"] is not None
            and "screenshot" in tool_call["result"]
            and int(tool_call["result"]["screenshot"].split(":")[-1].replace(">", ""))
            == image_index
        ):
            action = json.loads(tool_call["arguments"])
            image = plot_dot_to_image(prev_image, image, action)
            action = tool_call["arguments"]
            if "reasoning" in tool_call:
                reasoning = tool_call["reasoning"]
            current_tool_call_index = i
            break
    return image, action, reasoning


def load_dataset(dataset_path):
    dataset_path = os.path.join(SESSION_DIR, dataset_path)
    global \
        tool_calls_list, \
        images_list, \
        current_image_index, \
        dataset, \
        prompt_text, \
        prompt_text_idx
    dataset = datasets.load_from_disk(dataset_path)
    tool_calls_list = [json.loads(str(data)) for data in dataset["tool_calls"]]
    images_list = dataset["images"]
    image, action, reasoning = plot_image(
        images_list[0][current_image_index - 1] if current_image_index > 0 else None,
        images_list[0][current_image_index],
        tool_calls_list[0],
        current_image_index,
    )

    for i, tool_call in enumerate(tool_calls_list[0]):
        print(tool_call)
        if json.loads(tool_call["arguments"])["action"] == "submit":
            prompt_text = json.loads(tool_call["arguments"])["text"]
            prompt_text_idx = i
            break

    return (
        image,
        action,
        reasoning,
        prompt_text,
    )


def next_image(reasoning_text):
    global \
        tool_calls_list, \
        images_list, \
        current_image_index, \
        current_tool_call_index, \
        tool_calls_index

    # Save current reasoning if there are any updates
    if tool_calls_list is not None and reasoning_text.strip():
        tool_calls_list[tool_calls_index][current_tool_call_index]["reasoning"] = (
            reasoning_text
        )

    current_image_index += 1
    if current_image_index >= len(images_list[tool_calls_index]):
        gr.Info("No more next images!")
        current_image_index -= 1
    assert images_list is not None and tool_calls_list is not None
    image, action, reasoning = plot_image(
        images_list[0][current_image_index - 1] if current_image_index > 0 else None,
        images_list[tool_calls_index][current_image_index],
        tool_calls_list[tool_calls_index],
        current_image_index,
    )
    return image, action, reasoning


def prev_image(reasoning_text):
    global \
        tool_calls_list, \
        images_list, \
        current_image_index, \
        current_tool_call_index, \
        tool_calls_index

    # Save current reasoning if there are any updates
    if tool_calls_list is not None and reasoning_text.strip():
        tool_calls_list[tool_calls_index][current_tool_call_index]["reasoning"] = (
            reasoning_text
        )

    current_image_index -= 1
    if current_image_index < 0:
        gr.Info("No more previous images!")
        current_image_index += 1
    assert images_list is not None and tool_calls_list is not None
    image, action, reasoning = plot_image(
        images_list[0][current_image_index - 1] if current_image_index > 0 else None,
        images_list[tool_calls_index][current_image_index],
        tool_calls_list[tool_calls_index],
        current_image_index,
    )
    return image, action, reasoning


def edit_reasoning(reasoning):
    global tool_calls_list, current_tool_call_index
    assert tool_calls_list is not None
    tool_calls_list[0][current_tool_call_index]["reasoning"] = reasoning
    gr.Info("Editing is done!")


def save_dataset(dataset_path, prompt_text_box_value):
    dataset_path = os.path.join(SESSION_DIR, dataset_path)
    global tool_calls_list, images_list, dataset, prompt_text_idx
    assert (
        tool_calls_list is not None and images_list is not None and dataset is not None
    )

    if prompt_text_box_value != prompt_text:
        gr.Info("Prompt text is changed, saving...")
        prompt_tool_args = json.loads(tool_calls_list[0][prompt_text_idx]["arguments"])
        prompt_tool_args["text"] = prompt_text_box_value
        tool_calls_list[0][1]["arguments"] = json.dumps(prompt_tool_args)
    # Create a new dataset with updated tool_calls
    updated_data = {
        "tool_calls": [json.dumps(tool_call) for tool_call in tool_calls_list],
        "images": images_list,
    }
    updated_dataset = datasets.Dataset.from_dict(updated_data)
    updated_dataset.save_to_disk(dataset_path)
    gr.Info("Dataset saved successfully!")


def load_all_datasets():
    sessions = load_all_sessions()
    if sessions is None:
        return []
    return [session["source_folder"] for session in sessions]


def create_replay_gradio_ui():
    theme = gr_themes.Soft(
        primary_hue=gr_themes.colors.slate,
        secondary_hue=gr_themes.colors.gray,
        neutral_hue=gr_themes.colors.stone,
        text_size=gr_themes.sizes.text_md,
        radius_size=gr_themes.sizes.radius_lg,
    )
    with gr.Blocks(theme=theme) as app:
        with gr.Row():
            with gr.Column(scale=5):
                dataset_path = gr.Dropdown(
                    value=None, choices=load_all_datasets(), label="Datasets"
                )
            with gr.Column(scale=1):
                load_btn = gr.Button(value="Refresh Dataset", variant="secondary")

        with gr.Row():
            with gr.Column(scale=1):
                prompt_text_box = gr.Textbox(value="", label="Prompt", interactive=True)
            with gr.Column(scale=3):
                image = gr.Image(value=None, show_label=False)
        with gr.Row():
            action = gr.Textbox(value="", label="Action")
            reasoning = gr.Textbox(value="", label="Reasoning", interactive=True)

        with gr.Row():
            # reasoning_edit_btn = gr.Button(value="Edit Reasoning")
            prev_btn = gr.Button(value="Prev Step", variant="secondary")
            next_btn = gr.Button(value="Next Step", variant="secondary")
            save_btn = gr.Button(value="Save", variant="primary")
        load_btn.click(
            lambda: gr.Dropdown(choices=load_all_datasets(), value=""),
            inputs=[],
            outputs=[dataset_path],
        )
        dataset_path.change(
            load_dataset,
            inputs=dataset_path,
            outputs=[image, action, reasoning, prompt_text_box],
        )
        next_btn.click(
            next_image, inputs=[reasoning], outputs=[image, action, reasoning]
        )
        prev_btn.click(
            prev_image, inputs=[reasoning], outputs=[image, action, reasoning]
        )
        # reasoning_edit_btn.click(edit_reasoning, inputs=reasoning, outputs=[])
        save_btn.click(save_dataset, inputs=[dataset_path, prompt_text_box], outputs=[])
    return app


if __name__ == "__main__":
    app = create_replay_gradio_ui()
    app.launch(share=False)
