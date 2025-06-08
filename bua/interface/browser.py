"""Browser interface using Playwright for Chrome control."""

import asyncio
import tempfile
import time
from typing import Dict, List, Optional, Tuple

from loguru import logger
import platform
import os

try:
    from playwright.async_api import ViewportSize, async_playwright
except ImportError:
    raise ImportError(  # noqa: B904
        "Playwright is required for browser interface. "
        "Install with: pip install playwright && playwright install chrome"
    )

visualize_click_position_js = """([x, y]) => {
                // Create a temporary visual indicator
                const indicator = document.createElement('div');
                indicator.style.position = 'fixed'; // Changed to fixed to show in viewport coordinates
                indicator.style.left = `${x - 10}px`;
                indicator.style.top = `${y - 10}px`;
                indicator.style.width = '10px';
                indicator.style.height = '10px';
                indicator.style.borderRadius = '50%';
                indicator.style.backgroundColor = 'rgba(255, 0, 0, 0.5)';
                indicator.style.border = '2px solid red';
                indicator.style.zIndex = '10000';
                indicator.style.pointerEvents = 'none';
                document.body.appendChild(indicator);
                
                // Add a text label showing coordinates
                const label = document.createElement('div');
                label.style.position = 'fixed';
                label.style.left = `${x + 15}px`;
                label.style.top = `${y - 10}px`;
                label.style.backgroundColor = 'black';
                label.style.color = 'white';
                label.style.padding = '2px 5px';
                label.style.borderRadius = '3px';
                label.style.fontSize = '12px';
                label.style.zIndex = '10001';
                label.style.pointerEvents = 'none';
                label.textContent = `(${Math.round(x)}, ${Math.round(y)})`;
                document.body.appendChild(label);
                
                // Remove the indicators after a delay
                setTimeout(() => {
                    document.body.removeChild(indicator);
                    document.body.removeChild(label);
                }, 1000);
            }"""


class LocalBrowserInterface:
    """Interface for controlling Chrome browser using Playwright."""

    def __init__(
        self,
        user_data_dir: str | None = None,
        headless: bool = False,
        context_args: list[str] = [],
        viewport: ViewportSize | None = None,
        inactivity_timeout: int = 600,  # 10 minutes in seconds
    ):
        self._browser = None
        self._page = None
        self._playwright = None
        self._ready = False
        self.user_data_dir = user_data_dir
        self.context_args = context_args
        self.viewport = viewport
        self._redirect_in_progress = False
        self._redirect_promise = None
        
        # Inactivity timeout management
        self.inactivity_timeout = inactivity_timeout
        self._last_activity_time = time.time()
        self._timeout_task = None
        self._shutdown_due_to_inactivity = False
        
        if self._detect_headless_environment():
            self.headless = True
        else:
            self.headless = headless

    def _detect_headless_environment(self) -> bool:
        # Check if we're on Linux
        if platform.system().lower() != "linux":
            return False

        # Check for common headless environment indicators
        headless_indicators = [
            # No DISPLAY environment variable
            not os.environ.get("DISPLAY"),
            # Common CI environment variables
            os.environ.get("CI") == "true",
            os.environ.get("GITHUB_ACTIONS") == "true",
            os.environ.get("GITLAB_CI") == "true",
            os.environ.get("JENKINS_URL") is not None,
            # Docker environment
            os.path.exists("/.dockerenv"),
            # SSH session without X11 forwarding
            os.environ.get("SSH_CLIENT") and not os.environ.get("DISPLAY"),
        ]

        return any(headless_indicators)

    def _reset_activity_timer(self) -> None:
        """Reset the inactivity timer."""
        self._last_activity_time = time.time()

    async def _start_inactivity_monitor(self) -> None:
        """Start monitoring for inactivity and shutdown browser if timeout is reached."""
        if self.inactivity_timeout <= 0:
            return  # Timeout disabled
            
        async def monitor_inactivity():
            while self._ready and not self._shutdown_due_to_inactivity:
                try:
                    current_time = time.time()
                    time_since_last_activity = current_time - self._last_activity_time
                    
                    if time_since_last_activity >= self.inactivity_timeout:
                        logger.info(f"Browser inactive for {self.inactivity_timeout} seconds, shutting down to save resources")
                        self._shutdown_due_to_inactivity = True
                        await self._close_async()
                        break
                    
                    # Check every 30 seconds
                    await asyncio.sleep(30)
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.debug(f"Error in inactivity monitor: {e}")
                    await asyncio.sleep(30)
        
        self._timeout_task = asyncio.create_task(monitor_inactivity())

    def _stop_inactivity_monitor(self) -> None:
        """Stop the inactivity monitor."""
        if self._timeout_task and not self._timeout_task.done():
            self._timeout_task.cancel()
            self._timeout_task = None

    def is_shutdown_due_to_inactivity(self) -> bool:
        """Check if the browser was shutdown due to inactivity."""
        return self._shutdown_due_to_inactivity

    def get_time_since_last_activity(self) -> float:
        """Get the time in seconds since the last activity."""
        return time.time() - self._last_activity_time

    async def _launch_browser(self):
        """Launch Chrome browser using Playwright."""
        try:
            logger.info("Launching Chrome browser...")
            self._playwright = await async_playwright().start()

            if self.user_data_dir is None:
                temp_dir = tempfile.mkdtemp()
                self.user_data_dir = temp_dir
            self._browser = await self._playwright.chromium.launch_persistent_context(
                channel="chrome",
                user_data_dir=self.user_data_dir,
                headless=self.headless,  # Set to True for headless mode
                args=self.context_args,
                viewport=self.viewport,
            )

            # Create the very first page – this becomes the single "main" tab
            self._page = await self._browser.new_page()

            # Helper: redirect a newly opened page into the main one, then close
            async def _redirect_to_main(new_page):
                """Grab URL from popup and load it in the main page."""
                try:
                    # Mark redirect as in progress
                    self._redirect_in_progress = True
                    
                    # Wait until we have at least DOMContentLoaded so .url is final
                    await new_page.wait_for_load_state("domcontentloaded")
                    target_url = new_page.url

                    # Close the unwanted tab/window
                    await new_page.close()

                    # Navigate the primary page if we have something to show
                    if self._page and target_url and target_url != "about:blank":
                        await self._page.goto(target_url)
                        
                        # Fast redirect: only wait for DOM content to be loaded
                        # Full loading will be ensured in screenshot method if needed
                        try:
                            await self._page.wait_for_load_state("domcontentloaded", timeout=3000)
                        except Exception:
                            logger.debug("DOMContentLoaded timeout during redirect")
                        
                        logger.debug(f"Redirected popup to main tab: {target_url}")
                        
                    # Mark redirect as completed
                    self._redirect_in_progress = False
                    if self._redirect_promise:
                        self._redirect_promise.set_result(True)
                        self._redirect_promise = None
                        
                except Exception as e:
                    logger.warning(f"Popup redirection failed: {e}")
                    # Ensure we mark redirect as completed even on error
                    self._redirect_in_progress = False
                    if self._redirect_promise:
                        self._redirect_promise.set_exception(e)
                        self._redirect_promise = None
            
            # ---------------------------- Listeners ----------------------------
            # 1. Any page created at the *context* level (target="_blank", ctrl+click)
            self._browser.on("page", lambda p: asyncio.create_task(_redirect_to_main(p)))
            # 2. Any popup created by window.open() from *this* page
            self._page.on("popup", lambda p: asyncio.create_task(_redirect_to_main(p)))

            await self._page.goto("about:blank")

            self._ready = True  
            # Start inactivity monitoring
            await self._start_inactivity_monitor()
            logger.info("Chrome browser launched successfully")
        except Exception as e:
            logger.exception("")
            logger.error(f"Failed to launch browser: {e}")
            raise

    async def wait_for_ready(self, timeout: int = 60) -> None:
        """Wait for the browser interface to be ready."""
        if not self._ready:
            await self._launch_browser()

        # Verify the browser and page are ready
        if self._browser is None or self._page is None:
            raise RuntimeError("Browser failed to initialize")

        self._reset_activity_timer()
        logger.info("Browser interface is ready")

    def close(self) -> None:
        """Close the browser connection."""
        if self._ready:
            asyncio.create_task(self._close_async())

    async def _close_async(self) -> None:
        """Async close helper."""
        try:
            # Stop inactivity monitoring
            self._stop_inactivity_monitor()
            
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
            self._redirect_in_progress = False
            if self._redirect_promise:
                self._redirect_promise.cancel()
                self._redirect_promise = None

    def force_close(self) -> None:
        """Force close the browser connection."""
        self.close()

    async def visualize_click_position(self, x: int, y: int) -> None:
        """Visualize the click position."""
        if not self._page:
            raise RuntimeError("Browser not initialized")
        await self._page.evaluate(visualize_click_position_js, [x, y])
    
    async def _wait_for_redirect_completion(self, timeout: float = 5.0) -> None:
        """Wait for any ongoing redirect to complete."""
        if not self._redirect_in_progress:
            return
            
        if self._redirect_promise is None:
            self._redirect_promise = asyncio.Future()
            
        try:
            await asyncio.wait_for(self._redirect_promise, timeout=timeout)
        except asyncio.TimeoutError:
            logger.debug(f"Redirect completion wait timed out after {timeout}s")
            # Reset the state
            self._redirect_in_progress = False
            if self._redirect_promise:
                self._redirect_promise.cancel()
                self._redirect_promise = None
        except Exception as e:
            logger.debug(f"Redirect completion wait failed: {e}")
            # Reset the state
            self._redirect_in_progress = False
            self._redirect_promise = None

    # Mouse Actions
    async def left_click(
        self, x: Optional[int] = None, y: Optional[int] = None, visualize: bool = False
    ) -> None:
        """Perform a left click."""
        if not self._page:
            raise RuntimeError("Browser not initialized")
        assert x is not None and y is not None, "x and y must be provided"
        self._reset_activity_timer()
        if visualize:
            await self.visualize_click_position(x, y)
        await self._page.mouse.click(x, y)

    async def right_click(
        self, x: Optional[int] = None, y: Optional[int] = None
    ) -> None:
        """Perform a right click."""
        if not self._page:
            raise RuntimeError("Browser not initialized")
        assert x is not None and y is not None, "x and y must be provided"
        self._reset_activity_timer()
        await self._page.mouse.click(x, y, button="right")

    async def double_click(
        self, x: Optional[int] = None, y: Optional[int] = None, visualize: bool = False
    ) -> None:
        """Perform a double click."""
        if not self._page:
            raise RuntimeError("Browser not initialized")
        assert x is not None and y is not None, "x and y must be provided"
        self._reset_activity_timer()
        if visualize:
            await self.visualize_click_position(x, y)
        await self._page.mouse.dblclick(x, y)

    async def move_cursor(self, x: int, y: int) -> None:
        """Move the cursor to specified position."""
        if not self._page:
            raise RuntimeError("Browser not initialized")
        self._reset_activity_timer()
        await self._page.mouse.move(x, y)

    async def drag_to(
        self, x: int, y: int, button: str = "left", duration: float = 0.5
    ) -> None:
        """Drag from current position to specified coordinates."""
        if not self._page:
            raise RuntimeError("Browser not initialized")
        self._reset_activity_timer()

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
        self._reset_activity_timer()

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
        self._reset_activity_timer()
        await self._page.keyboard.type(text)

    async def press_key(self, key: str) -> None:
        """Press a single key."""
        if not self._page:
            raise RuntimeError("Browser not initialized")
        self._reset_activity_timer()

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
        self._reset_activity_timer()

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
    async def scroll_down(
        self, x: Optional[int] = None, y: Optional[int] = None, clicks: int = 1
    ) -> None:
        """Scroll down."""
        if not self._page:
            raise RuntimeError("Browser not initialized")
        self._reset_activity_timer()
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

    async def scroll_up(
        self, x: Optional[int] = None, y: Optional[int] = None, clicks: int = 1
    ) -> None:
        """Scroll up."""
        if not self._page:
            raise RuntimeError("Browser not initialized")
        self._reset_activity_timer()

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
        self._reset_activity_timer()

        # Ensure any ongoing redirects are completed before taking screenshot
        await self._wait_for_redirect_completion()
        
        # Ensure the page is fully loaded before taking screenshot
        try:
            # Wait for all resources including images, stylesheets, scripts
            await self._page.wait_for_load_state("load", timeout=8000)
        except Exception:
            try:
                # Fallback: wait for network to be idle
                await self._page.wait_for_load_state("networkidle", timeout=5000)
            except Exception:
                try:
                    # Final fallback: at least DOM should be loaded
                    await self._page.wait_for_load_state("domcontentloaded", timeout=3000)
                except Exception:
                    logger.debug("All load state waits timed out, proceeding with screenshot")
        
        # Additional wait for dynamic content and rendering
        await asyncio.sleep(1)
        
        screenshot_bytes = await self._page.screenshot()
        return screenshot_bytes

    async def get_screen_size(self) -> Dict[str, int]:
        """Get the screen dimensions.

        Returns:
            Dict with 'width' and 'height' keys
        """
        if not self._page:
            raise RuntimeError("Browser not initialized")
        self._reset_activity_timer()
        viewport = self._page.viewport_size
        if viewport is None:
            # Default viewport size if not set
            return {"width": 1024, "height": 768}
        return {"width": viewport["width"], "height": viewport["height"]}

    async def copy_to_clipboard(self) -> str:
        """Get clipboard content."""
        if not self._page:
            raise RuntimeError("Browser not initialized")

        self._reset_activity_timer()
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
        self._reset_activity_timer()

        # Use JavaScript to set clipboard
        await self._page.evaluate(
            """
            (text) => {
                navigator.clipboard.writeText(text);
            }
        """,
            text,
        )

    async def go_to_url(self, url: str) -> None:
        """Go to a URL."""
        if not self._page:
            raise RuntimeError("Browser not initialized")
        self._reset_activity_timer()
        print(f"Going to URL: {url}")
        await self._page.goto(url)
        await self._page.wait_for_load_state("domcontentloaded")

    async def triple_click(self, x: int, y: int, visualize: bool = False) -> None:
        """Perform a triple click."""
        if not self._page:
            raise RuntimeError("Browser not initialized")
        self._reset_activity_timer()
        if visualize:
            await self.visualize_click_position(x, y)
        await self._page.mouse.click(x, y, click_count=3)
