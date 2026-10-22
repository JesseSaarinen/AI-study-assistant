"""
Record the README demo GIF and screenshots by driving a running app in a browser:
load the sample notes, ask a question and a follow-up, and take a short quiz.

Needs Playwright and ffmpeg, which the app itself does not:

    pip install playwright && playwright install chromium
    streamlit run app.py --client.toolbarMode minimal &
    python scripts/record_demo.py --url http://localhost:8501

The answers come from whichever chat model the app is configured with. Waits for
the chat model are cut short in the GIF, so a slow provider does not make it longer.
"""

import argparse
import csv
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path

from playwright.sync_api import Locator, Page, ViewportSize, expect, sync_playwright

REPO_ROOT = Path(__file__).resolve().parent.parent
VIEWPORT: ViewportSize = {"width": 1152, "height": 768}

FIRST_QUESTION = "Why is appending to a dynamic array O(1) amortised?"
FOLLOW_UP = "Is it faster than a linked list in practice?"
QUIZ_TOPIC = "sorting algorithms"
SOURCES_LABEL = re.compile(r"^Sources: \d+ retrieved")
SCORE_LABEL = re.compile(r"^Score: \d+ / \d+")
FLASHCARDS_LABEL = re.compile(r"^Download as flashcards")

# Generous, because the chat client retries rate-limited and timed-out requests.
RESPONSE_TIMEOUT_MS = 240_000

# Seconds of each wait for the chat model that the GIF keeps: the start shows
# the spinner and the end shows the answer arriving.
KEEP_WAIT_START = 1.0
KEEP_WAIT_END = 1.5

VIDEO_FPS = 25


@dataclass
class Timeline:
    """
    The parts of the recording to cut from the GIF, in seconds from its start.
    """

    started: float = field(default_factory=time.perf_counter)
    cuts: list[tuple[float, float]] = field(default_factory=list)

    def now(self) -> float:
        return time.perf_counter() - self.started

    def cut_until_now(self) -> None:
        self.cuts.append((0.0, self.now()))

    @contextmanager
    def waiting_for_model(self) -> Iterator[None]:
        """
        Cut the middle of a wait for the chat model, whose length depends on the
        provider rather than on the app.
        """

        start = self.now()
        yield
        end = self.now()

        if end - start > KEEP_WAIT_START + KEEP_WAIT_END:
            self.cuts.append((start + KEEP_WAIT_START, end - KEEP_WAIT_END))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0].strip())
    parser.add_argument("--url", default="http://localhost:8501")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "docs")
    parser.add_argument(
        "--speed",
        type=float,
        default=1.5,
        help="Playback speed of the GIF compared with the recording.",
    )
    parser.add_argument("--width", type=int, default=960, help="Width of the GIF in pixels.")
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument(
        "--chromium-path",
        default=None,
        help="Use this Chromium binary instead of the one Playwright installed.",
    )

    return parser.parse_args()


def wait_until_idle(page: Page, timeout_ms: int = RESPONSE_TIMEOUT_MS) -> None:
    """
    Wait until Streamlit has finished rerunning the script.
    """

    page.wait_for_timeout(500)
    page.locator('[data-testid="stStatusWidget"]').wait_for(state="hidden", timeout=timeout_ms)


def pause(page: Page, seconds: float) -> None:
    page.wait_for_timeout(int(seconds * 1000))


def scroll_to_bottom(page: Page) -> None:
    page.mouse.move(VIEWPORT["width"] * 0.6, VIEWPORT["height"] * 0.5)
    page.mouse.wheel(0, 5000)
    pause(page, 0.4)


def scroll_to_top_of(element: Locator) -> None:
    """
    Scroll the page so the element sits at the top of the window, clear of
    Streamlit's header.
    """

    element.evaluate(
        "element => {"
        " element.style.scrollMarginTop = '64px';"
        " element.scrollIntoView({block: 'start'});"
        " }"
    )


def collapse_sidebar(page: Page) -> None:
    """
    Hide the settings sidebar so the recording has room for the chat.
    """

    page.get_by_test_id("stSidebar").hover()
    page.get_by_test_id("stSidebarCollapseButton").locator("button").click()
    pause(page, 0.5)


def read_correct_options(page: Page) -> list[str]:
    """
    Read the correct option of each quiz question from the flashcard download.

    The button gets a click event rather than a real click, so the page does
    not scroll down to it.
    """

    with page.expect_download() as download:
        page.get_by_role("button", name=FLASHCARDS_LABEL).dispatch_event("click")

    with open(download.value.path(), newline="", encoding="utf-8") as file:
        # The back of each card starts with the correct option.
        return [back.split("\n\n", 1)[0] for _question, back in csv.reader(file)]


def find_option(radio_group: Locator, option: str) -> int:
    """
    Return the index of the label closest to an option's text. Labels are
    rendered as Markdown, so an exact comparison could miss.
    """

    labels = [
        re.sub(r"^[A-Z]\. ", "", label) for label in radio_group.locator("label").all_inner_texts()
    ]

    return max(
        range(len(labels)),
        key=lambda index: SequenceMatcher(None, labels[index], option).ratio(),
    )


def record(page: Page, url: str, output_dir: Path, timeline: Timeline) -> None:
    """
    Drive the app through the demo, noting in the timeline what to cut.
    """

    page.goto(url)
    page.get_by_role("button", name="Try with sample notes").wait_for(timeout=60_000)
    collapse_sidebar(page)
    timeline.cut_until_now()
    pause(page, 1.5)

    page.get_by_role("button", name="Try with sample notes").click()
    page.get_by_text("Processing complete.").wait_for(timeout=RESPONSE_TIMEOUT_MS)
    pause(page, 1.5)

    page.get_by_role("tab", name="3. Ask questions").click()
    pause(page, 1.5)
    page.get_by_role("button", name=FIRST_QUESTION).click()
    pause(page, 1.0)
    scroll_to_bottom(page)

    with timeline.waiting_for_model():
        page.get_by_text(SOURCES_LABEL).first.wait_for(timeout=RESPONSE_TIMEOUT_MS)

    wait_until_idle(page)
    scroll_to_bottom(page)
    pause(page, 2.5)

    chat_input = page.get_by_placeholder("Ask about your notes, or ask a follow-up question")
    chat_input.click()
    chat_input.press_sequentially(FOLLOW_UP, delay=35)
    pause(page, 0.5)
    chat_input.press("Enter")

    with timeline.waiting_for_model():
        page.get_by_text("Searched the notes for").wait_for(timeout=RESPONSE_TIMEOUT_MS)

    scroll_to_bottom(page)
    wait_until_idle(page)
    scroll_to_bottom(page)
    pause(page, 1.5)

    # A follow-up the notes cannot answer gets no sources, which would leave
    # the recording without its sources panel.
    sources = page.get_by_text(SOURCES_LABEL)
    expect(sources, "The follow-up question was not answered from the notes.").to_have_count(2)
    sources.last.click()
    pause(page, 0.6)

    # Show the passage the answer cites rather than the first one retrieved.
    cited_tabs = page.get_by_role("tab", name=re.compile("✅"))

    if cited_tabs.count():
        cited_tabs.first.click()
        pause(page, 0.6)

    scroll_to_bottom(page)
    pause(page, 2.5)

    # Frame the follow-up from the question down to its sources.
    scroll_to_top_of(page.get_by_test_id("stChatMessage").nth(-2))
    pause(page, 1.5)
    page.screenshot(path=output_dir / "demo-chat.png")

    page.get_by_role("tab", name="4. Practice").click()
    pause(page, 1.0)
    topic = page.get_by_label("Topic (optional)")
    topic.click()
    topic.press_sequentially(QUIZ_TOPIC, delay=35)

    slider = page.get_by_role("slider")
    slider.focus()
    for _ in range(slider_steps_to(page, 3)):
        slider.press("ArrowLeft")
    pause(page, 0.5)

    page.get_by_role("button", name="Generate quiz").click()

    with timeline.waiting_for_model():
        page.get_by_role("button", name="Check answers").wait_for(timeout=RESPONSE_TIMEOUT_MS)

    wait_until_idle(page)
    scroll_to_top_of(page.get_by_test_id("stForm").last)
    pause(page, 1.5)

    # Get every question right except the second, so the recording shows both
    # kinds of feedback. The radio inputs are visually hidden, so click the
    # option's label the way a user would.
    correct_options = read_correct_options(page)

    for number, radio_group in enumerate(page.get_by_role("radiogroup").all()):
        choice = find_option(radio_group, correct_options[number])

        if number == 1:
            choice = (choice + 1) % radio_group.locator("label").count()

        radio_group.locator("label").nth(choice).click()
        expect(radio_group.get_by_role("radio").nth(choice)).to_be_checked()
        pause(page, 0.6)

    page.get_by_role("button", name="Check answers").click()
    page.get_by_text(SCORE_LABEL).wait_for(timeout=RESPONSE_TIMEOUT_MS)
    wait_until_idle(page)
    scroll_to_top_of(page.get_by_text(SCORE_LABEL))
    pause(page, 2.5)
    page.screenshot(path=output_dir / "demo-quiz.png")


def slider_steps_to(page: Page, target: int) -> int:
    current = int(page.get_by_role("slider").input_value() or target)
    return max(current - target, 0)


def convert_to_gif(
    video: Path,
    gif: Path,
    speed: float,
    width: int,
    fps: int,
    cuts: list[tuple[float, float]],
) -> None:
    if shutil.which("ffmpeg") is None:
        sys.exit("ffmpeg is needed to turn the recording into a GIF.")

    # Drop the frames inside the cuts and close the gaps they leave. Then
    # mpdecimate drops frames that differ from the last one kept only by video
    # compression noise, and fps fills the gaps with exact copies, which the GIF
    # stores almost for free.
    dropped = "+".join(f"between(t,{start:.2f},{end:.2f})" for start, end in cuts) or "0"
    filters = (
        f"fps={VIDEO_FPS},select='not({dropped})',setpts=N/FRAME_RATE/TB/{speed},"
        f"fps={fps},scale={width}:-1:flags=lanczos,mpdecimate,fps={fps},"
        "split[a][b];[a]palettegen=max_colors=128:stats_mode=diff[palette];"
        "[b][palette]paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle"
    )

    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(video),
            "-vf",
            filters,
            "-loop",
            "0",
            str(gif),
        ],
        check=True,
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as video_dir, sync_playwright() as playwright:
        browser = playwright.chromium.launch(executable_path=args.chromium_path)
        context = browser.new_context(
            viewport=VIEWPORT,
            record_video_dir=video_dir,
            record_video_size=VIEWPORT,
        )
        page = context.new_page()
        timeline = Timeline()

        try:
            record(page, args.url, args.output_dir, timeline)
        finally:
            context.close()
            browser.close()

        if page.video is None:
            sys.exit("Playwright did not record a video.")

        gif = args.output_dir / "demo.gif"
        convert_to_gif(
            video=Path(page.video.path()),
            gif=gif,
            speed=args.speed,
            width=args.width,
            fps=args.fps,
            cuts=timeline.cuts,
        )

    cut_seconds = sum(end - start for start, end in timeline.cuts)
    print(f"Cut {cut_seconds:.0f} s of loading and waiting from the recording.")
    print(f"Wrote {gif} ({gif.stat().st_size / 1_000_000:.1f} MB)")


if __name__ == "__main__":
    main()
