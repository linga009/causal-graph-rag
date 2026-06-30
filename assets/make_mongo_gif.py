"""
make_mongo_gif.py — render assets/demo_mongo.gif headlessly (Pillow, no recorder).

    pip install Pillow
    python assets/make_mongo_gif.py
"""
import os
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "demo_mongo.gif")
W, PAD, LH, TOP, FS = 1200, 26, 29, 64, 20


def _font(bold=False):
    names = (["consolab.ttf", "DejaVuSansMono-Bold.ttf"] if bold
             else ["consola.ttf", "DejaVuSansMono.ttf"])
    for r in (r"C:\Windows\Fonts", "/usr/share/fonts/truetype/dejavu", "/Library/Fonts"):
        for n in names:
            p = os.path.join(r, n)
            if os.path.exists(p):
                return ImageFont.truetype(p, FS)
    return ImageFont.load_default()


FONT, BOLD = _font(), _font(True)
BG, BAR = (13, 17, 23), (32, 38, 48)
DEF, DIM, GRN, CYN, YEL, MONGO = (
    (201, 209, 217), (110, 118, 129), (63, 185, 80), (88, 166, 255),
    (240, 200, 90), (0, 237, 100))   # MongoDB green


def R(t, c=DEF, b=False):
    return (t, c, b)


LINES = [
    [R("Causal Graph RAG on ", DEF, True), R("MongoDB", MONGO, True), R("  — traverse cause→effect with $graphLookup", DIM)],
    [],
    [R('# "The reactor overheated. The coolant valve failed … the outage', DIM)],
    [R('   disrupted hospital operations."', DIM)],
    [],
    [R("✓ wrote 4 causal edges as documents to ", GRN), R("'causal_edges'", CYN)],
    [],
    [R("# native traversal — runs IN MongoDB:", DIM)],
    [R("db.causal_edges.aggregate([", CYN)],
    [R('  { $match: { cause: "reactor overheated" } },', CYN)],
    [R('  { $graphLookup: { from:"causal_edges", startWith:"$effect",', CYN)],
    [R('      connectFromField:"effect", connectToField:"cause", maxDepth:5 } } ])', CYN)],
    [],
    [R("Downstream impact of ", DEF, True), R("'reactor overheated'", DEF, True), R("   (hop depth):", DIM)],
    [R("  1↳ coolant valve", YEL)],
    [R("  2↳ emergency shutdown", YEL)],
    [R("  3↳ power outage", YEL)],
    [R("  4↳ hospital operations", YEL)],
    [R("  $graphLookup in 2.4 ms", DIM)],
    [],
    [R("reactor overheated -/-> coolant valve -> emergency shutdown", CYN)],
    [R("  -> power outage -/-> hospital operations", CYN)],
    [],
    [R("→ graph traversal + causal reasoning, natively on MongoDB.", MONGO, True)],
]

STEPS = [(1, 700), (4, 800), (6, 900), (8, 300), (12, 1600), (13, 500),
         (14, 250), (15, 250), (16, 250), (17, 250), (19, 1500),
         (22, 1600), (24, 2800)]
H = TOP + PAD + len(LINES) * LH + PAD


def render(n):
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, W, 40], fill=BAR)
    for i, c in enumerate([(255, 95, 86), (255, 189, 46), (39, 201, 63)]):
        d.ellipse([20 + i * 26, 14, 32 + i * 26, 26], fill=c)
    d.text((W // 2 - 90, 11), "causal-rag · mongodb", font=FONT, fill=DIM)
    y = TOP + PAD
    for ln in LINES[:n]:
        x = PAD
        for t, c, b in ln:
            d.text((x, y), t, font=(BOLD if b else FONT), fill=c)
            x += FONT.getlength(t)
        y += LH
    return img


def main():
    frames = [render(s) for s, _ in STEPS]
    frames[0].save(OUT, save_all=True, append_images=frames[1:],
                   duration=[d for _, d in STEPS], loop=0, optimize=True, disposal=1)
    print(f"wrote {OUT}  ({round(os.path.getsize(OUT)/1024,1)} KB, {len(frames)} frames, {W}x{H})")


if __name__ == "__main__":
    main()
