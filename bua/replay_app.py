import datasets
import json
import gradio as gr
from PIL import Image, ImageDraw

tool_calls_list: list[list[dict]] | None = None
images_list: list[list[Image.Image]] | None = None

current_image_index = 0
def plot_image(image, tool_calls, image_index):
    action = None
    for tool_call in tool_calls:
        if 'result' in tool_call and tool_call['result'] is not None and 'screenshot' in tool_call['result'] and int(tool_call['result']['screenshot'].split(':')[-1].replace('>', '')) == image_index:
            action = json.loads(tool_call['arguments'])
            if "x" in action and "y" in action:
                image = image.convert("RGBA")
                draw = ImageDraw.Draw(image)
                draw.rectangle([action["x"], action["y"], action["x"] + 10, action["y"] + 10], outline="red", width=2)
            action = json.dumps(action, ensure_ascii=False)
            break
    return image, action

def load_dataset(dataset_path):
    global tool_calls_list, images_list, current_image_index
    dataset = datasets.load_from_disk(dataset_path)
    tool_calls_list = [json.loads(str(data)) for data in dataset['tool_calls']]
    images_list = dataset['images']
    image, action = plot_image(images_list[0][current_image_index], tool_calls_list[0], current_image_index)
    return gr.Dropdown(choices=[i for i in range(len(tool_calls_list))], value=0, interactive=True), image, action



def next_image(tool_calls_index):
    global tool_calls_list, images_list, current_image_index

    current_image_index += 1
    assert images_list is not None and tool_calls_list is not None
    image, action = plot_image(images_list[tool_calls_index][current_image_index], tool_calls_list[tool_calls_index], current_image_index)
    return image, action

def prev_image(tool_calls_index):
    global tool_calls_list, images_list, current_image_index

    current_image_index -= 1
    assert images_list is not None and tool_calls_list is not None
    image, action = plot_image(images_list[tool_calls_index][current_image_index], tool_calls_list[tool_calls_index], current_image_index)
    return image, action

def create_replay_gradio_ui():
    with gr.Blocks() as app:
        gr.Markdown("Replay App")
        with gr.Row():
            dataset_path = gr.Textbox(label="Dataset Path")
            load_btn = gr.Button(value="Load Dataset")

        with gr.Row():
            tool_calls_index = gr.Dropdown(choices=[], label="Tool Calls Index")

        with gr.Row():
            image = gr.Image(value=None)
        with gr.Row():
            action = gr.Textbox(value="", label="Action")

        with gr.Row():
            prev_btn = gr.Button(value="Prev Step")
            next_btn = gr.Button(value="Next Step")
            

        load_btn.click(load_dataset, inputs=dataset_path, outputs=[tool_calls_index, image, action])
        next_btn.click(next_image, inputs=tool_calls_index, outputs=[image, action])
        prev_btn.click(prev_image, inputs=tool_calls_index, outputs=[image, action])

    return app


if __name__ == "__main__":
    app = create_replay_gradio_ui()
    app.launch(share=False)



