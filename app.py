import gradio as gr

from bua.gradio.anno_app import create_gradio_ui
from bua.gradio.replay_app import create_replay_gradio_ui

main_app = create_gradio_ui()
replay_app = create_replay_gradio_ui()

with gr.Blocks() as demo:
    main_app.render()


with demo.route("Reasoning Annotation", path='/reasoning'):
    replay_app.render()

if __name__ == "__main__":
    demo.launch(share=False)
