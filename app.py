import gradio as gr
from gradio.themes import ThemeClass

from bua.gradio.anno_app import create_gradio_ui
from bua.gradio.replay_app import create_replay_gradio_ui
import subprocess
import sys


theme = ThemeClass.load("./static/theme_taithrah_minimal.json")


def ensure_playwright_chrome():
    try:
        subprocess.run(["playwright", "--version"], capture_output=True, check=True)

        result = subprocess.run(
            ["playwright", "install", "--dry-run"],
            capture_output=True,
            text=True,
            check=False,
        )

        if result.returncode == 0 and "chrome" not in result.stdout.lower():
            print("Chrome for Playwright is already available.")
            return

        install_result = subprocess.run(
            ["playwright", "install", "chrome"],
            capture_output=True,
            text=True,
            check=False,
        )

        if install_result.returncode == 0:
            print("Chrome for Playwright installed successfully.")
        elif "already installed" in install_result.stderr.lower():
            print("Chrome is already installed and ready to use.")
        else:
            print(f"Note: {install_result.stderr}")
            print("Proceeding with existing Chrome installation...")

    except subprocess.CalledProcessError as e:
        print(f"Error running playwright command: {e}")
        sys.exit(1)
    except FileNotFoundError:
        print(
            "Playwright not found. Please install it first with: pip install playwright"
        )
        sys.exit(1)


ensure_playwright_chrome()


main_app = create_gradio_ui()
replay_app = create_replay_gradio_ui()

with gr.Blocks(theme=theme) as demo:
    main_app.render()


with demo.route("Reasoning Annotation", path="/reasoning"):
    replay_app.render()

if __name__ == "__main__":
    demo.launch(share=False)
