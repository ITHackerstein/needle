# AI DISCLAIMER: this file was written with the assistance of Claude
# (Anthropic), directed by the author.

# NOTE: inlines docs/figures/*.png into docs/slides.html as data URIs, because the
# artifact host blocks every external request. Run after changing slides.html.
import base64
import pathlib
import re

HERE = pathlib.Path(__file__).parent
TEMPLATE = HERE / "slides.html"
OUT = HERE / "slides.build.html"
PRINT = HERE / "slides.print.html"


def data_uri(name):
    raw = (HERE / "figures" / (name + ".png")).read_bytes()
    return "data:image/png;base64," + base64.b64encode(raw).decode("ascii")


def main():
    html = TEMPLATE.read_text()
    used = set(re.findall(r"\{\{fig:([a-z_]+)\}\}", html))
    for name in sorted(used):
        html = html.replace("{{fig:" + name + "}}", data_uri(name))
    OUT.write_text(html)

    # NOTE: the artifact host supplies the document skeleton, so slides.build.html
    # has none. A local file needs its own or the browser renders in quirks mode;
    # this copy is the one to open and Ctrl+P into a PDF.
    head, body = html.split('<div id="viewport">', 1)
    PRINT.write_text(
        '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        + head + '</head>\n<body>\n<div id="viewport">' + body + "\n</body>\n</html>\n"
    )
    print("wrote", OUT, "({} KiB, {} figures)".format(len(html) // 1024, len(used)))
    print("wrote", PRINT, "(standalone, printable)")


if __name__ == "__main__":
    main()
