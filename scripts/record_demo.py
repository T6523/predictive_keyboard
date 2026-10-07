"""Records the README's demo GIF: types a sentence into the live predictive-keyboard
page one letter at a time, pauses on each new word's first letter long enough to show
the 3 suggestion chips populate, and clicks a chip twice to show the insert-and-continue
loop. Saves a .webm via Playwright's own video recording (crop/convert to GIF is a
separate ffmpeg step, not done here).

Usage:
    python3 record_demo.py --url https://t6523.github.io/predictive_keyboard/ \
        --out-dir ../assets/_record
"""
import argparse
import json
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

# Typed as (text_so_far_delta, pause_after_ms). Each entry is appended to the textarea's
# value one event at a time (matching how a real keystroke fires 'input'), with a pause
# after the FIRST letter of each new word so the suggestion chips have time to render and
# be visible on camera. Context is intentionally short -- the model only ever looks at
# the last 3 completed words (MAX_CONTEXT in app.js), so "stage , which" is enough to
# reproduce the same real top-3 result a much longer lead-in sentence would ("...tuesday
# ' s 11th stage , which w" -> was/will/would, verified earlier this session), without
# spending GIF seconds typing words the lookup never uses.
SCRIPT = [
    ("stage , which ", 300),
    ("w", 1500),        # real 3-suggestion case: chips show "was" / "will" / "would"
    ("[CLICK:1]", 1200),  # click chip 1 ("will") -> inserts "will "
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--out-dir", default="../assets/_record")
    ap.add_argument("--width", type=int, default=1000)
    ap.add_argument("--height", type=int, default=750)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            viewport={"width": args.width, "height": args.height},
            record_video_dir=str(out_dir),
            record_video_size={"width": args.width, "height": args.height},
        )
        page = context.new_page()
        page.goto(args.url, wait_until="load")

        # Readiness signal: app.js sets typer.disabled = false only after the ~10MB
        # model finishes loading -- not a fixed sleep.
        page.wait_for_function("() => !document.getElementById('typer').disabled", timeout=30000)
        ready_at = time.time()
        print("model ready, waiting 0.5s before starting")
        time.sleep(0.5)

        box = page.locator(".hero-inner").bounding_box()
        print("app container bbox:", json.dumps(box))

        typer = page.locator("#typer")
        typer.click()

        for chunk, pause_ms in SCRIPT:
            if chunk.startswith("[CLICK:"):
                idx = int(chunk[len("[CLICK:"):-1])
                page.locator(".chip").nth(idx).click()
            else:
                current = typer.input_value()
                for ch in chunk:
                    typer.evaluate(
                        "(el, ch) => { el.value += ch; el.dispatchEvent(new Event('input', {bubbles:true})); }",
                        ch,
                    )
                    time.sleep(0.07)  # ~80-150ms/char human-typing pace
            time.sleep(pause_ms / 1000)

        print("holding final state 1.5s")
        time.sleep(1.5)

        context.close()  # flushes the .webm
        browser.close()

        webm_files = sorted(out_dir.glob("*.webm"), key=lambda f: f.stat().st_mtime)
        print("recorded ->", webm_files[-1])
        print("ready_at (unix):", ready_at)


if __name__ == "__main__":
    main()
