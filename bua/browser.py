"""Browser interface using Playwright for Chrome control."""

import asyncio
import io
import json
import base64
from typing import Any, Dict, List, Optional, Tuple
from PIL import Image
from loguru import logger

class BrowserInterface:
    """Interface for controlling Chrome browser using Playwright."""

    def __init__(
        self,
    ):
        self._browser = None
        self._page = None
        self._playwright = None
        self._ready = False

    async def _launch_browser(self):
        """Launch Chrome browser using Playwright."""
        try:
            from playwright.async_api import async_playwright

            logger.info("Launching Chrome browser...")
            self._playwright = await async_playwright().start()

            self._browser = await self._playwright.chromium.launch_persistent_context(
                channel="chrome",
                headless=False,  # Set to True for headless mode
                viewport={"width": 1280, "height": 800},
            )

            # Create a new page with viewport
            self._page = await self._browser.new_page()

            await self._page.goto("about:blank")

            logger.info("Chrome browser launched successfully")
            self._ready = True

        except ImportError:
            raise ImportError(
                "Playwright is required for browser interface. "
                "Install with: pip install playwright && playwright install chromium"
            )
        except Exception as e:
            from loguru import logger
            logger.exception('')
            logger.error(f"Failed to launch browser: {e}")
            raise

    async def wait_for_ready(self, timeout: int = 60) -> None:
        """Wait for the browser interface to be ready."""
        if not self._ready:
            await self._launch_browser()

        # Verify the browser and page are ready
        if self._browser is None or self._page is None:
            raise RuntimeError("Browser failed to initialize")

        logger.info("Browser interface is ready")

    def close(self) -> None:
        """Close the browser connection."""
        if self._ready:
            asyncio.create_task(self._close_async())

    async def _close_async(self) -> None:
        """Async close helper."""
        try:
            if self._page:
                await self._page.close()
            if self._browser:
                await self._browser.close()
            if self._playwright:
                await self._playwright.stop()
        except Exception as e:
            logger.debug(f"Error during cleanup: {e}")
        finally:
            self._ready = False
            self._page = None
            self._browser = None
            self._playwright = None

    def force_close(self) -> None:
        """Force close the browser connection."""
        self.close()

    # Mouse Actions
    async def left_click(self, x: Optional[int] = None, y: Optional[int] = None) -> None:
        """Perform a left click."""
        if not self._page:
            raise RuntimeError("Browser not initialized")
        assert x is not None and y is not None, "x and y must be provided"

        await self._page.mouse.click(x, y)

    async def right_click(self, x: Optional[int] = None, y: Optional[int] = None) -> None:
        """Perform a right click."""
        if not self._page:
            raise RuntimeError("Browser not initialized")
        assert x is not None and y is not None, "x and y must be provided"
        await self._page.mouse.click(x, y, button="right")

    async def double_click(self, x: Optional[int] = None, y: Optional[int] = None) -> None:
        """Perform a double click."""
        if not self._page:
            raise RuntimeError("Browser not initialized")
        assert x is not None and y is not None, "x and y must be provided"
        await self._page.mouse.dblclick(x, y)

    async def move_cursor(self, x: int, y: int) -> None:
        """Move the cursor to specified position."""
        if not self._page:
            raise RuntimeError("Browser not initialized")

        await self._page.mouse.move(x, y)

    async def drag_to(self, x: int, y: int, button: str = "left", duration: float = 0.5) -> None:
        """Drag from current position to specified coordinates."""
        if not self._page:
            raise RuntimeError("Browser not initialized")

        # Get current mouse position (assume center if not tracked)
        current_x, current_y = 512, 384

        await self._page.mouse.move(current_x, current_y)
        await self._page.mouse.down()
        await asyncio.sleep(duration)
        await self._page.mouse.move(x, y)
        await self._page.mouse.up()

    async def drag(
        self, path: List[Tuple[int, int]], button: str = "left", duration: float = 0.5
    ) -> None:
        """Drag the cursor along a path of coordinates."""
        if not self._page or not path:
            raise RuntimeError("Browser not initialized or empty path")

        # Start at first position
        start_x, start_y = path[0]
        await self._page.mouse.move(start_x, start_y)
        await self._page.mouse.down()

        # Calculate delay between each step
        step_delay = duration / len(path) if len(path) > 1 else 0

        # Move through the path
        for x, y in path[1:]:
            await self._page.mouse.move(x, y)
            await asyncio.sleep(step_delay)

        await self._page.mouse.up()

    # Keyboard Actions
    async def type_text(self, text: str) -> None:
        """Type the specified text."""
        if not self._page:
            raise RuntimeError("Browser not initialized")

        await self._page.keyboard.type(text)

    async def press_key(self, key: str) -> None:
        """Press a single key."""
        if not self._page:
            raise RuntimeError("Browser not initialized")

        # Map common key names to Playwright key names
        key_mapping = {
            "enter": "Enter",
            "return": "Enter",
            "tab": "Tab",
            "space": "Space",
            "escape": "Escape",
            "backspace": "Backspace",
            "delete": "Delete",
            "up": "ArrowUp",
            "down": "ArrowDown",
            "left": "ArrowLeft",
            "right": "ArrowRight",
            "home": "Home",
            "end": "End",
            "page_up": "PageUp",
            "page_down": "PageDown",
            "f1": "F1",
            "f2": "F2",
            "f3": "F3",
            "f4": "F4",
            "f5": "F5",
            "f6": "F6",
            "f7": "F7",
            "f8": "F8",
            "f9": "F9",
            "f10": "F10",
            "f11": "F11",
            "f12": "F12",
        }

        playwright_key = key_mapping.get(key.lower(), key)
        await self._page.keyboard.press(playwright_key)

    async def hotkey(self, *keys: str) -> None:
        """Press multiple keys simultaneously."""
        if not self._page:
            raise RuntimeError("Browser not initialized")

        # Convert keys to Playwright format
        key_mapping = {
            "cmd": "Meta",
            "ctrl": "Control",
            "alt": "Alt",
            "shift": "Shift",
            "meta": "Meta",
            "control": "Control",
        }

        playwright_keys = []
        for key in keys:
            mapped_key = key_mapping.get(key.lower(), key)
            playwright_keys.append(mapped_key)

        # Press all keys down
        for key in playwright_keys:
            await self._page.keyboard.down(key)

        # Release all keys (in reverse order)
        for key in reversed(playwright_keys):
            await self._page.keyboard.up(key)

    # Scrolling Actions
    async def scroll_down(self, x: Optional[int]=None, y: Optional[int]=None, clicks: int = 1) -> None:
        """Scroll down."""
        if not self._page:
            raise RuntimeError("Browser not initialized")
        viewport = self._page.viewport_size
        if viewport is None:
            viewport_height = 768  # Default height
        else:
            viewport_height = viewport["height"]
        if x is None or y is None:
            # Get viewport height and calculate scroll distance (0.5 viewport height)


            scroll_distance = viewport_height // 2

            for _ in range(clicks):
                await self._page.mouse.wheel(0, scroll_distance)
        else:
            # Scroll at position (x, y) for dropdown menus
            await self._page.mouse.move(x, y)
            # Default mouse wheel scroll distance is typically 120 units per click
            scroll_distance = 0.1 * viewport_height
            for _ in range(clicks):
                await self._page.mouse.wheel(0, scroll_distance)


    async def scroll_up(self, x: Optional[int]=None, y: Optional[int]=None, clicks: int = 1) -> None:
        """Scroll up."""
        if not self._page:
            raise RuntimeError("Browser not initialized")

        viewport = self._page.viewport_size
        if viewport is None:
            viewport_height = 768  # Default height
        else:
            viewport_height = viewport["height"]
        # Get viewport height and calculate scroll distance (0.5 viewport height)
        if x is None or y is None:


            scroll_distance = viewport_height // 2

            for _ in range(clicks):
                await self._page.mouse.wheel(0, -scroll_distance)
        else:
            await self._page.mouse.move(x, y)
            scroll_distance = 0.1 * viewport_height
            for _ in range(clicks):
                await self._page.mouse.wheel(0, -scroll_distance)

    # Screen Actions
    async def screenshot(self) -> bytes:
        """Take a screenshot.

        Returns:
            Raw bytes of the screenshot image
        """
        if not self._page:
            raise RuntimeError("Browser not initialized")

        screenshot_bytes = await self._page.screenshot()
        return screenshot_bytes

    async def get_screen_size(self) -> Dict[str, int]:
        """Get the screen dimensions.

        Returns:
            Dict with 'width' and 'height' keys
        """
        if not self._page:
            raise RuntimeError("Browser not initialized")

        viewport = self._page.viewport_size
        if viewport is None:
            # Default viewport size if not set
            return {"width": 1024, "height": 768}
        return {"width": viewport["width"], "height": viewport["height"]}

    async def get_cursor_position(self) -> Dict[str, int]:
        """Get current cursor position."""
        # Playwright doesn't provide direct access to cursor position
        # Return center of viewport as fallback
        viewport = await self.get_screen_size()
        return {"x": viewport["width"] // 2, "y": viewport["height"] // 2}

    # Clipboard Actions
    async def copy_to_clipboard(self) -> str:
        """Get clipboard content."""
        if not self._page:
            raise RuntimeError("Browser not initialized")

        # Use JavaScript to access clipboard
        try:
            clipboard_content = await self._page.evaluate("""
                () => {
                    return navigator.clipboard.readText();
                }
            """)
            return clipboard_content
        except Exception:
            # Fallback: try to get selected text
            try:
                selected_text = await self._page.evaluate("""
                    () => {
                        return window.getSelection().toString();
                    }
                """)
                return selected_text
            except Exception:
                return ""

    async def set_clipboard(self, text: str) -> None:
        """Set clipboard content."""
        if not self._page:
            raise RuntimeError("Browser not initialized")

        # Use JavaScript to set clipboard
        await self._page.evaluate(
            """
            (text) => {
                navigator.clipboard.writeText(text);
            }
        """,
            text,
        )

    # File System Actions
    async def file_exists(self, path: str) -> bool:
        """Check if file exists."""
        # For browser interface, this doesn't make sense
        # Return False as browser doesn't have direct file system access
        return False

    async def directory_exists(self, path: str) -> bool:
        """Check if directory exists."""
        # For browser interface, this doesn't make sense
        return False

    async def run_command(self, command: str) -> Tuple[str, str]:
        """Run shell command."""
        # For browser interface, we can't run shell commands
        # But we can navigate to URLs if the command looks like a URL
        if command.startswith(("http://", "https://", "www.")):
            try:
                if not self._page:
                    raise RuntimeError("Browser not initialized")
                if not command.startswith(("http://", "https://")):
                    command = "https://" + command
                await self._page.goto(command)
                return f"Navigated to {command}", ""
            except Exception as e:
                return "", f"Failed to navigate: {str(e)}"
        else:
            return "", "Shell commands not supported in browser interface"

    # Accessibility Actions
    async def get_accessibility_tree(self) -> Dict:
        """Get the accessibility tree of the current screen."""
        if not self._page:
            raise RuntimeError("Browser not initialized")

        try:
            # Get accessibility tree using Playwright's accessibility API
            accessibility_tree = await self._page.accessibility.snapshot()
            return accessibility_tree or {}
        except Exception as e:
            logger.debug(f"Failed to get accessibility tree: {e}")
            return {}

    async def to_screen_coordinates(self, x: float, y: float) -> tuple[float, float]:
        """Convert screenshot coordinates to screen coordinates.

        Args:
            x: X coordinate in screenshot space
            y: Y coordinate in screenshot space

        Returns:
            tuple[float, float]: (x, y) coordinates in screen space
        """
        # For browser interface, screenshot and screen coordinates are the same
        return (x, y)

    async def to_screenshot_coordinates(self, x: float, y: float) -> tuple[float, float]:
        """Convert screen coordinates to screenshot coordinates.

        Args:
            x: X coordinate in screen space
            y: Y coordinate in screen space

        Returns:
            tuple[float, float]: (x, y) coordinates in screenshot space
        """
        # For browser interface, screenshot and screen coordinates are the same
        return (x, y)

    # Browser-specific methods
    async def navigate_to(self, url: str) -> None:
        """Navigate to a specific URL."""
        if not self._page:
            raise RuntimeError("Browser not initialized")

        await self._page.goto(url)

    async def get_current_url(self) -> str:
        """Get the current page URL."""
        if not self._page:
            raise RuntimeError("Browser not initialized")

        return self._page.url

    async def get_page_title(self) -> str:
        """Get the current page title."""
        if not self._page:
            raise RuntimeError("Browser not initialized")

        return await self._page.title()

    async def wait_for_selector(self, selector: str, timeout: int = 5000) -> None:
        """Wait for an element with the given selector to appear."""
        if not self._page:
            raise RuntimeError("Browser not initialized")

        await self._page.wait_for_selector(selector, timeout=timeout)

    async def click_selector(self, selector: str) -> None:
        """Click on an element with the given selector."""
        if not self._page:
            raise RuntimeError("Browser not initialized")

        await self._page.click(selector)

    async def fill_selector(self, selector: str, text: str) -> None:
        """Fill an input element with the given selector."""
        if not self._page:
            raise RuntimeError("Browser not initialized")

        await self._page.fill(selector, text)

    async def get_element_text(self, selector: str) -> str:
        """Get text content of an element."""
        if not self._page:
            raise RuntimeError("Browser not initialized")

        return await self._page.text_content(selector) or ""

    async def evaluate_javascript(self, script: str) -> Any:
        """Execute JavaScript on the current page."""
        if not self._page:
            raise RuntimeError("Browser not initialized")

        return await self._page.evaluate(script)

    async def go_to_url(self, url: str) -> None:
        """Go to a URL."""
        if not self._page:
            raise RuntimeError("Browser not initialized")
        print(f"Going to URL: {url}")
        await self._page.goto(url)
        await self._page.wait_for_load_state("domcontentloaded")

    async def triple_click(self, x: int, y: int) -> None:
        """Perform a triple click."""
        if not self._page:
            raise RuntimeError("Browser not initialized")

        await self._page.mouse.click(x, y, click_count=3)
