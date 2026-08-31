# AI DISCLAIMER: this file was written with the assistance of Claude
# (Anthropic), directed by the author.

# NOTE: renders docs/slides.print.html to docs/slides.pdf, one 13.333x7.5in page
# per slide. Chromium prints vector text through the deck's @media print rules;
# Firefox has no headless print-to-pdf, so that fallback screenshots each slide
# and binds the frames, which costs selectable text and sharpness on zoom.
import pathlib
import shutil
import subprocess
import sys
import tempfile

HERE = pathlib.Path(__file__).parent
PAGE = HERE / "slides.print.html"
OUT = HERE / "slides.pdf"
W, H = 1280, 720


def with_chromium(exe, tmp):
    subprocess.run(
        [exe, "--headless", "--disable-gpu", "--no-sandbox", "--no-pdf-header-footer",
         "--virtual-time-budget=20000", "--user-data-dir=" + str(tmp / "cr"),
         "--print-to-pdf=" + str(OUT), PAGE.as_uri()],
        check=True, capture_output=True,
    )


def with_firefox(exe, tmp):
    from PIL import Image

    count = PAGE.read_text().count('class="slide')
    shot_page = tmp / "frames.html"
    shot_page.write_text(PAGE.read_text().replace(
        "</head>",
        "<style>.slide{transition:none!important}"
        "#chrome,#hint,.nav{display:none!important}</style></head>", 1))
    profile = tmp / "profile"
    profile.mkdir()

    frames = []
    for n in range(1, count + 1):
        shot = tmp / ("%03d.png" % n)
        subprocess.run(
            [exe, "--profile", str(profile), "--headless", "--window-size", "%d,%d" % (W, H),
             "--screenshot", str(shot), "%s#%d" % (shot_page.as_uri(), n)],
            check=True, capture_output=True, env={"MOZ_HEADLESS": "1", "HOME": str(tmp)},
        )
        if not shot.exists():
            sys.exit("no frame for slide %d" % n)
        frames.append(Image.open(shot).convert("RGB"))
    frames[0].save(OUT, save_all=True, append_images=frames[1:], resolution=96.0)


def main():
    if not PAGE.exists():
        sys.exit("run build_slides.py first")
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="needle-slides-"))
    try:
        chromium = shutil.which("chromium") or shutil.which("chromium-browser") \
            or shutil.which("google-chrome-stable")
        if chromium:
            with_chromium(chromium, tmp)
            how = "chromium, vector"
        else:
            firefox = shutil.which("firefox")
            if not firefox:
                sys.exit("need chromium or firefox")
            with_firefox(firefox, tmp)
            how = "firefox screenshots, raster"
        print("wrote %s (%d KiB, %s)" % (OUT, OUT.stat().st_size // 1024, how))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
