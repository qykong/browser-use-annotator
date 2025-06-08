import json
import os
import zipfile
import tempfile
from pathlib import Path

import datasets
from PIL import Image, ImageDraw
from joblib import Parallel, delayed

import gradio as gr
import gradio.themes as gr_themes
from bua.gradio.constants import generate_session_id, get_session_dir
from bua.gradio.utils import load_all_sessions
from bua.llm.prompts import REASONING_PROMPT_TEMPLATE
from bua.llm.utils import call_openai_vlm

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


def process_single_action(action_data, goal, current_images, openai_config):
    """Process a single action for reasoning generation"""
    tool_call_idx, tool_call, image_idx, action_history = action_data
    
    try:
        action_args = json.loads(tool_call["arguments"])
        action_description = f"{action_args.get('action', 'unknown')}"
        if "x" in action_args and "y" in action_args:
            action_description += (
                f" at coordinates ({action_args['x']}, {action_args['y']})"
            )
        if "text" in action_args:
            action_description += f" with text: '{action_args['text']}'"

        before_image = current_images[image_idx - 1] if image_idx > 0 else None
        after_image = (
            current_images[image_idx] if image_idx < len(current_images) else None
        )
        before_image, after_image = plot_dot_to_image(
            before_image, after_image, action_args, combine_images=False
        )

        reasoning = call_openai_vlm(
            REASONING_PROMPT_TEMPLATE.render(
                goal=goal, action=action_description, action_history=action_history
            ),
            [before_image, after_image],
            api_key=openai_config.get("api_key", ""),
            base_url=openai_config.get("base_url", "https://api.openai.com/v1"),
            model=openai_config.get("model", "gpt-4o"),
        )
        reasoning = reasoning.replace("Reasoning:", "").strip()
        
        return tool_call_idx, reasoning, None

    except Exception as e:
        return tool_call_idx, f"Auto-annotation failed: {str(e)}", str(e)


def auto_annotate_all_actions(
    session_id,
    dataset_path,
    openai_config,
    cur_image,
    cur_action,
    cur_reasoning,
    cur_prompt_text,
    progress=gr.Progress(),
):
    """Auto-annotate all actions in the dataset using VLM with concurrent processing"""
    if not dataset_path:
        gr.Warning("Please select a dataset first!")
        return cur_image, cur_action, cur_reasoning, cur_prompt_text

    if not openai_config.get("api_key", "").strip():
        gr.Warning("Please provide an OpenAI API key!")
        return cur_image, cur_action, cur_reasoning, cur_prompt_text

    if not openai_config.get("model", "").strip():
        gr.Warning("Please specify a model to use!")
        return cur_image, cur_action, cur_reasoning, cur_prompt_text

    # Initialize session if needed
    initialize_replay_session(session_id)

    # Load dataset if not already loaded
    if tool_calls_list[session_id] is None:
        load_dataset(session_id, dataset_path)

    if tool_calls_list[session_id] is None or images_list[session_id] is None:
        gr.Warning("Failed to load dataset!")
        return cur_image, cur_action, cur_reasoning, cur_prompt_text

    # Get the current tool calls and images
    current_tool_calls = tool_calls_list[session_id][tool_calls_index[session_id]]
    current_images = images_list[session_id][tool_calls_index[session_id]]
    goal = prompt_text[session_id]

    # Filter out actions that don't have screenshots (like submit actions)
    actions_to_annotate = []
    action_history = []
    
    for i, tool_call in enumerate(current_tool_calls):
        if i <= prompt_text_idx[session_id]:
            continue
        if (
            "result" in tool_call
            and tool_call["result"] is not None
            and "screenshot" in tool_call["result"]
        ):
            # Extract image index from result
            try:
                screenshot_ref = tool_call["result"]["screenshot"]
                image_idx = int(screenshot_ref.split(":")[-1].replace(">", ""))
                
                # Skip if reasoning already exists
                if not tool_call.get("reasoning", "").strip():
                    actions_to_annotate.append((i, tool_call, image_idx, action_history.copy()))
                
                # Build action history for context
                action_args = json.loads(tool_call["arguments"])
                action_desc = f"{action_args.get('action', 'unknown')}"
                if "x" in action_args and "y" in action_args:
                    action_desc += f" at coordinates ({action_args['x']}, {action_args['y']})"
                if "text" in action_args:
                    action_desc += f" with text: '{action_args['text']}'"
                action_history.append(action_desc)
                
            except (ValueError, IndexError):
                continue

    if not actions_to_annotate:
        gr.Info("No actions found that need annotation!")
        return cur_image, cur_action, cur_reasoning, cur_prompt_text

    progress(0, desc="Starting auto-annotation...")

    # Process actions in parallel with progress tracking
    n_jobs = min(openai_config.get("max_workers", 1), len(actions_to_annotate))
    
    try:
        # Create a generator that yields delayed tasks while updating progress
        def create_delayed_tasks():
            for idx, action_data in enumerate(actions_to_annotate):
                # Update progress as we submit each task
                progress_percent = idx / len(actions_to_annotate)
                progress(progress_percent, desc=f"Submitting action {idx + 1}/{len(actions_to_annotate)} for processing...")
                yield delayed(process_single_action)(action_data, goal, current_images, openai_config)
        
        # Process all actions in parallel
        progress(0.1, desc="Processing actions in parallel...")
        results = Parallel(n_jobs=n_jobs, backend="threading")(create_delayed_tasks())

        # Update tool calls with results
        progress(0.9, desc="Collecting results...")
        successful_annotations = 0
        for tool_call_idx, reasoning, error in results:
            tool_calls_list[session_id][tool_calls_index[session_id]][tool_call_idx]["reasoning"] = reasoning
            if error is None:
                successful_annotations += 1
            else:
                print(f"Error processing action {tool_call_idx}: {error}")

        progress(1.0, desc="Auto-annotation completed!")
        gr.Info(f"Successfully auto-annotated {successful_annotations}/{len(actions_to_annotate)} actions!")

    except Exception as e:
        progress(1.0, desc="Auto-annotation failed!")
        gr.Warning(f"Auto-annotation failed: {str(e)}")

    # Return updated current display
    cur_reasoning = tool_calls_list[session_id][tool_calls_index[session_id]][
        current_tool_call_index[session_id]
    ]["reasoning"]
    return cur_image, cur_action, cur_reasoning, cur_prompt_text


def plot_dot_to_image(prev_image, result_image, action, combine_images=True):
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

    if combine_images:
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
    else:
        return prev_image, result_image


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


def create_dataset_download(session_id, dataset_path, prompt_text_box_value):
    """Create a zip file of the selected dataset for download"""
    if not dataset_path:
        gr.Warning("Please select a dataset first!")
        return None
    save_dataset(session_id, dataset_path, prompt_text_box_value)
    
    try:
        # Get the full dataset path
        full_dataset_path = os.path.join(get_session_dir(session_id), dataset_path)
        
        if not os.path.exists(full_dataset_path):
            gr.Warning("Dataset path does not exist!")
            return None
        
        # Create a temporary zip file
        temp_dir = tempfile.mkdtemp()
        zip_filename = f"{dataset_path.replace('/', '_')}.zip"
        zip_path = os.path.join(temp_dir, zip_filename)
        
        # Create the zip file
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            dataset_path_obj = Path(full_dataset_path)
            
            # Add all files in the dataset directory to the zip
            for file_path in dataset_path_obj.rglob('*'):
                if file_path.is_file():
                    # Calculate the relative path for the zip archive
                    relative_path = file_path.relative_to(dataset_path_obj.parent)
                    zipf.write(file_path, relative_path)
        
        gr.Info(f"Dataset '{dataset_path}' has been prepared for download!")
        return zip_path
        
    except Exception as e:
        gr.Warning(f"Failed to create download file: {str(e)}")
        return None


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

        # Consolidated OpenAI configuration state
        openai_config_state = gr.BrowserState(
            {
                "api_key": "",
                "base_url": "https://api.openai.com/v1",
                "model": "gpt-4o",
                "max_workers": 1
            },
            storage_key="openai_config",
            secret="openai_config",
        )

        gr.Markdown(f"# Reasoning Annotation")
        gr.Markdown(
            "🌟 **[Star us on GitHub](https://github.com/qykong/browser-use-annotator)** to support this project!"
        )

        # Add API Configuration Section
        with gr.Accordion("OpenAI API Configuration", open=False):
            with gr.Row():
                api_key_input = gr.Textbox(
                    label="OpenAI API Key",
                    type="password",
                    placeholder="Enter your OpenAI API key...",
                    interactive=True,
                )
                base_url_input = gr.Textbox(
                    label="Base URL (Optional)",
                    placeholder="https://api.openai.com/v1",
                    value="https://api.openai.com/v1",
                    interactive=True,
                )
            with gr.Row():
                model_input = gr.Textbox(
                    label="Model",
                    placeholder="gpt-4o",
                    value="gpt-4o",
                    interactive=True,
                )
                max_workers_input = gr.Number(
                    label="Max Concurrent Requests",
                    value=1,
                    minimum=1,
                    maximum=20,
                    step=1,
                    interactive=True,
                )

        with gr.Row():
            with gr.Column(scale=5):
                dataset_path = gr.Dropdown(choices=[], label="Datasets")
            with gr.Column(scale=1):
                load_btn = gr.Button(value="Refresh Dataset", variant="secondary")
                auto_annotate_btn = gr.Button(
                    value="Auto Annotate with VLM", variant="secondary"
                )

        with gr.Row():
            with gr.Column(scale=1):
                prompt_text_box = gr.Textbox(value="", label="Prompt", interactive=True)
            with gr.Column(scale=3):
                image = gr.Image(value=None, show_label=False)
        with gr.Row():
            action = gr.Textbox(value="", label="Action")
            reasoning = gr.Textbox(value="", label="Reasoning", interactive=True)

        with gr.Row():
            prev_btn = gr.Button(value="Prev Step", variant="secondary")
            next_btn = gr.Button(value="Next Step", variant="secondary")
            save_btn = gr.Button(value="Save", variant="primary")
            download_btn = gr.DownloadButton(
                    label="Download Annotated Dataset", 
                    variant="primary",
                    visible=False
                )

        # Session initialization on app load
        @app.load(
            inputs=[session_state, openai_config_state],
            outputs=[
                session_state,
                dataset_path,
                api_key_input,
                base_url_input,
                model_input,
                max_workers_input,
            ],
        )
        def initialize_session_on_load(session_id, openai_config):
            """Initialize session when app loads"""
            if session_id == "":
                session_id = generate_session_id()
            print(f"Initializing replay session {session_id}")
            initialize_replay_session(session_id)

            # Load saved OpenAI configuration
            api_key = openai_config.get("api_key", "")
            base_url = openai_config.get("base_url", "https://api.openai.com/v1")
            model = openai_config.get("model", "gpt-4o")
            max_workers = openai_config.get("max_workers", 3)

            return (
                session_id,
                gr.Dropdown(choices=load_all_datasets(session_id)),
                api_key,
                base_url,
                model,
                max_workers,
            )

        # Save OpenAI configuration when changed
        def save_openai_config(openai_config, api_key, base_url, model, max_workers):
            """Save OpenAI configuration to browser state"""
            updated_config = {
                "api_key": api_key,
                "base_url": base_url if base_url.strip() else "https://api.openai.com/v1",
                "model": model,
                "max_workers": int(max_workers)
            }
            return updated_config

        # Update config on any input change
        for input_component in [api_key_input, base_url_input, model_input, max_workers_input]:
            input_component.change(
                save_openai_config,
                inputs=[openai_config_state, api_key_input, base_url_input, model_input, max_workers_input],
                outputs=[openai_config_state],
            )

        # Function to update download button visibility
        def update_download_button_visibility(dataset_path_value):
            """Show download button when a dataset is selected"""
            return gr.DownloadButton(visible=bool(dataset_path_value))

        load_btn.click(
            lambda session_id: gr.Dropdown(choices=load_all_datasets(session_id)),
            inputs=[session_state],
            outputs=[dataset_path],
        )
        
        dataset_path.change(
            load_dataset,
            inputs=[session_state, dataset_path],
            outputs=[image, action, reasoning, prompt_text_box],
        ).then(
            update_download_button_visibility,
            inputs=[dataset_path],
            outputs=[download_btn]
        )

        # Download button click handler
        download_btn.click(
            create_dataset_download,
            inputs=[session_state, dataset_path, prompt_text_box],
            outputs=[download_btn]
        )

        # Auto-annotate button with consolidated config
        auto_annotate_btn.click(
            auto_annotate_all_actions,
            inputs=[
                session_state,
                dataset_path,
                openai_config_state,
                image,
                action,
                reasoning,
                prompt_text_box,
            ],
            outputs=[image, action, reasoning, prompt_text_box],
        )

        next_btn.click(
            next_image,
            inputs=[session_state, reasoning],
            outputs=[image, action, reasoning],
        )
        prev_btn.click(
            prev_image,
            inputs=[session_state, reasoning],
            outputs=[image, action, reasoning],
        )
        save_btn.click(
            save_dataset,
            inputs=[session_state, dataset_path, prompt_text_box],
            outputs=[],
        )
    return app


if __name__ == "__main__":
    app = create_replay_gradio_ui()
    app.launch(share=False)
