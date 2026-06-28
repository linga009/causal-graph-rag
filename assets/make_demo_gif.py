"""
make_demo_gif.py — render assets/demo.gif headlessly with Pillow (no terminal
recorder needed). Produced the committed GIF; re-run to regenerate.

    pip install Pillow
    python assets/make_demo_gif.py
"""
import os
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "demo.gif")

W, PAD, LH, TOP, FS = 1140, 26, 30, 64, 21


def _font(bold=False):
    names = (["consolab.ttf", "DejaVuSansMono-Bold.ttf", "courbd.ttf"] if bold
             else ["consola.ttf", "DejaVuSansMono.ttf", "cour.ttf"])
    roots = [r"C:\Windows\Fonts", "/usr/share/fonts/truetype/dejavu",
             "/Library/Fonts", "/usr/share/fonts"]
    for r in roots:
        for n in names:
            p = os.path.join(r, n)
            if os.path.exists(p):
                return ImageFont.truetype(p, FS)
    return ImageFont.load_default()


FONT, TITLEF = _font(), _font(bold=True)

BG, BAR = (13, 17, 23), (32, 38, 48)
DEF, DIM, MAG, GRN, CYN, YEL, WHT = (
    (201, 209, 217), (110, 118, 129), (210, 168, 255), (63, 185, 80),
    (88, 166, 255), (240, 200, 90), (245, 245, 250))


def R(text, color=DEF, bold=False):
    return (text, color, bold)


LINES = [
    [R("Causal Graph RAG", MAG, True), R("  — why · what-if · root-cause, by traversing cause→effect chains", DIM)],
    [],
    [R("# incident report", DIM)],
    [R('"The reactor overheated. The coolant valve failed. This triggered an', DIM)],
    [R(' emergency shutdown. The shutdown caused a power outage. The power', DIM)],
    [R(' outage disrupted hospital operations."', DIM)],
    [],
    [R("$ ", DIM), R("causal-rag ingest incident.md --save graph.pkl", WHT)],
    [R("  ✓ 4 causal edges, 5 nodes  ", GRN), R("(spaCy / rules — no LLM)", DIM)],
    [],
    [R("$ ", DIM), R('causal-rag rootcause graph.pkl "hospital operations"', WHT)],
    [R("  reactor overheated -/->(implicit_trigger) coolant valve ->(trigger)", CYN)],
    [R("  emergency shutdown ->(cause) power outage -/->(disrupt) hospital ops", CYN)],
    [R("  traced in 0.11 ms  ", DIM), R("· no LLM, no embedding search", DIM)],
    [],
    [R("→ flat RAG can't connect these at any price — the answer is 4 hops away.", YEL, True)],
    [],
    [R("  pip install causal-graph-rag", GRN), R("   ·   github.com/linga009/causal-graph-rag", DIM)],
]

STEPS = [(1, 700), (2, 150), (3, 500), (6, 900), (7, 200), (8, 700), (9, 900),
         (10, 200), (11, 700), (13, 1500), (14, 1200), (15, 200), (16, 1500), (18, 2800)]

H = TOP + PAD + len(LINES) * LH + PAD


def render(n):
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, W, 40], fill=BAR)
    for i, c in enumerate([(255, 95, 86), (255, 189, 46), (39, 201, 63)]):
        d.ellipse([20 + i * 26, 14, 32 + i * 26, 26], fill=c)
    d.text((W // 2 - 70, 11), "causal-rag — demo", font=FONT, fill=DIM)
    y = TOP + PAD
    for line in LINES[:n]:
        x = PAD
        for text, color, bold in line:
            d.text((x, y), text, font=(TITLEF if bold else FONT), fill=color)
            x += FONT.getlength(text)
        y += LH
    return img


def main():
    frames = [render(s) for s, _ in STEPS]
    frames[0].save(OUT, save_all=True, append_images=frames[1:],
                   duration=[d for _, d in STEPS], loop=0, optimize=True, disposal=1)
    print(f"wrote {OUT}  ({round(os.path.getsize(OUT)/1024,1)} KB, {len(frames)} frames, {W}x{H})")


if __name__ == "__main__":
    main()
