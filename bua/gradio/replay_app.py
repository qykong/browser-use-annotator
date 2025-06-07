import json
import os

import datasets
from PIL import Image, ImageDraw

import gradio as gr
import gradio.themes as gr_themes
from bua.gradio.constants import generate_session_id, get_session_dir
from bua.gradio.utils import load_all_sessions

# Session-specific state dictionaries (indexed by session_id)
tool_calls_list = {}  # session_id -> list[list[dict]] | None
images_list = {}  # session_id -> list[list[Image.Image]] | None
dataset = {}  # session_id -> datasets.Dataset | datasets.DatasetDict | None
current_image_index = {}  # session_id -> int
current_tool_call_index = {}  # session_id -> int
prompt_text = {}  # session_id -> str
tool_calls_index = {}  # session_id -> int
prompt_text_idx = {}  # session_id -> int


def initialize_replay_session(session_id):
    """Initialize all session-specific state for replay app"""
    if session_id not in tool_calls_list:
        tool_calls_list[session_id] = None
        images_list[session_id] = None
        dataset[session_id] = None
        current_image_index[session_id] = 2
        current_tool_call_index[session_id] = 2
        prompt_text[session_id] = ""
        tool_calls_index[session_id] = 0
        prompt_text_idx[session_id] = 0


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

    draw_combined.text(
        (width // 2 - 60, 5), "Before Action", fill="black", font_size=20
    )
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
        outline="black",
        width=border_width,
    )
    # Border for after image
    draw_combined.rectangle(
        [width, label_height, width * 2 - 1, label_height + height - 1],
        outline="black",
        width=border_width,
    )

    return combined


def plot_image(prev_image, image, tool_calls, image_index, session_id):
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
            current_tool_call_index[session_id] = i
            break
    return image, action, reasoning


def load_dataset(session_id, dataset_path):
    dataset_path = os.path.join(get_session_dir(session_id), dataset_path)

    # Initialize session if needed
    initialize_replay_session(session_id)

    dataset[session_id] = datasets.load_from_disk(dataset_path)
    tool_calls_list[session_id] = [
        json.loads(str(data)) for data in dataset[session_id]["tool_calls"]
    ]
    images_list[session_id] = dataset[session_id]["images"]

    image, action, reasoning = plot_image(
        images_list[session_id][0][current_image_index[session_id] - 1]
        if current_image_index[session_id] > 0
        else None,
        images_list[session_id][0][current_image_index[session_id]],
        tool_calls_list[session_id][0],
        current_image_index[session_id],
        session_id,
    )

    for i, tool_call in enumerate(tool_calls_list[session_id][0]):
        print(tool_call)
        if json.loads(tool_call["arguments"])["action"] == "submit":
            prompt_text[session_id] = json.loads(tool_call["arguments"])["text"]
            prompt_text_idx[session_id] = i
            break

    return (
        image,
        action,
        reasoning,
        prompt_text[session_id],
    )


def next_image(session_id, reasoning_text):
    # Initialize session if needed
    initialize_replay_session(session_id)

    # Save current reasoning if there are any updates
    if tool_calls_list[session_id] is not None and reasoning_text.strip():
        tool_calls_list[session_id][tool_calls_index[session_id]][
            current_tool_call_index[session_id]
        ]["reasoning"] = reasoning_text

    current_image_index[session_id] += 1
    if current_image_index[session_id] >= len(
        images_list[session_id][tool_calls_index[session_id]]
    ):
        gr.Info("No more next images!")
        current_image_index[session_id] -= 1
    assert (
        images_list[session_id] is not None and tool_calls_list[session_id] is not None
    )
    image, action, reasoning = plot_image(
        images_list[session_id][0][current_image_index[session_id] - 1]
        if current_image_index[session_id] > 0
        else None,
        images_list[session_id][tool_calls_index[session_id]][
            current_image_index[session_id]
        ],
        tool_calls_list[session_id][tool_calls_index[session_id]],
        current_image_index[session_id],
        session_id,
    )
    return image, action, reasoning


def prev_image(session_id, reasoning_text):
    # Initialize session if needed
    initialize_replay_session(session_id)

    # Save current reasoning if there are any updates
    if tool_calls_list[session_id] is not None and reasoning_text.strip():
        tool_calls_list[session_id][tool_calls_index[session_id]][
            current_tool_call_index[session_id]
        ]["reasoning"] = reasoning_text

    current_image_index[session_id] -= 1
    if current_image_index[session_id] < 0:
        gr.Info("No more previous images!")
        current_image_index[session_id] += 1
    assert (
        images_list[session_id] is not None and tool_calls_list[session_id] is not None
    )
    image, action, reasoning = plot_image(
        images_list[session_id][0][current_image_index[session_id] - 1]
        if current_image_index[session_id] > 0
        else None,
        images_list[session_id][tool_calls_index[session_id]][
            current_image_index[session_id]
        ],
        tool_calls_list[session_id][tool_calls_index[session_id]],
        current_image_index[session_id],
        session_id,
    )
    return image, action, reasoning


def edit_reasoning(session_id, reasoning):
    # Initialize session if needed
    initialize_replay_session(session_id)

    assert tool_calls_list[session_id] is not None
    tool_calls_list[session_id][0][current_tool_call_index[session_id]]["reasoning"] = (
        reasoning
    )
    gr.Info("Editing is done!")


def save_dataset(session_id, dataset_path, prompt_text_box_value):
    dataset_path = os.path.join(get_session_dir(session_id), dataset_path)

    # Initialize session if needed
    initialize_replay_session(session_id)

    assert (
        tool_calls_list[session_id] is not None
        and images_list[session_id] is not None
        and dataset[session_id] is not None
    )

    if prompt_text_box_value != prompt_text[session_id]:
        gr.Info("Prompt text is changed, saving...")
        prompt_tool_args = json.loads(
            tool_calls_list[session_id][0][prompt_text_idx[session_id]]["arguments"]
        )
        prompt_tool_args["text"] = prompt_text_box_value
        tool_calls_list[session_id][0][1]["arguments"] = json.dumps(prompt_tool_args)
    # Create a new dataset with updated tool_calls
    updated_data = {
        "tool_calls": [
            json.dumps(tool_call) for tool_call in tool_calls_list[session_id]
        ],
        "images": images_list[session_id],
    }
    updated_dataset = datasets.Dataset.from_dict(updated_data)
    updated_dataset.save_to_disk(dataset_path)
    gr.Info("Dataset saved successfully!")


def load_all_datasets(session_id):
    sessions = load_all_sessions(session_id=session_id)
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
        session_state = gr.BrowserState(
            "", storage_key="annotation_app", secret="annotation_app"
        )

        gr.Markdown(f"# Reasoning Annotation")
        gr.Markdown("🌟 **[Star us on GitHub](https://github.com/qykong/browser-use-annotator)** to support this project!")

        with gr.Row():
            with gr.Column(scale=5):
                dataset_path = gr.Dropdown(choices=[], label="Datasets")
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

        # Session initialization on app load
        @app.load(inputs=[session_state], outputs=[session_state, dataset_path])
        def initialize_session_on_load(session_id):
            """Initialize session when app loads"""
            if session_id == "":
                session_id = generate_session_id()
            print(f"Initializing replay session {session_id}")
            initialize_replay_session(session_id)
            return session_id, gr.Dropdown(choices=load_all_datasets(session_id))

        # Wrapper functions to handle session state
        def load_dataset_wrapper(session_id, dataset_path):
            if not dataset_path:
                return None, "", "", ""
            return load_dataset(session_id, dataset_path)

        def next_image_wrapper(session_id, reasoning_text):
            return next_image(session_id, reasoning_text)

        def prev_image_wrapper(session_id, reasoning_text):
            return prev_image(session_id, reasoning_text)

        def save_dataset_wrapper(session_id, dataset_path, prompt_text_box_value):
            if not dataset_path:
                return
            return save_dataset(session_id, dataset_path, prompt_text_box_value)

        load_btn.click(
            lambda session_id: gr.Dropdown(choices=load_all_datasets(session_id)),
            inputs=[session_state],
            outputs=[dataset_path],
        )
        dataset_path.change(
            load_dataset_wrapper,
            inputs=[session_state, dataset_path],
            outputs=[image, action, reasoning, prompt_text_box],
        )
        next_btn.click(
            next_image_wrapper,
            inputs=[session_state, reasoning],
            outputs=[image, action, reasoning],
        )
        prev_btn.click(
            prev_image_wrapper,
            inputs=[session_state, reasoning],
            outputs=[image, action, reasoning],
        )
        # reasoning_edit_btn.click(edit_reasoning, inputs=reasoning, outputs=[])
        save_btn.click(
            save_dataset_wrapper,
            inputs=[session_state, dataset_path, prompt_text_box],
            outputs=[],
        )
    return app


if __name__ == "__main__":
    app = create_replay_gradio_ui()
    app.launch(share=False)
