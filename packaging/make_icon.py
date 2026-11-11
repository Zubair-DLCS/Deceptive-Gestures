"""Draws the Deceptive Gestures icon (a half-closed eye on an orange tile) -> .ico and .png."""
from pathlib import Path

from PIL import Image, ImageDraw

S = 1024
img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
d = ImageDraw.Draw(img)
d.rounded_rectangle([40, 40, S - 40, S - 40], radius=210, fill=(220, 96, 46))
W, H = 800, 440                              # almond eye: the overlap of two circles
R = ((W / 2) ** 2 + (H / 2) ** 2) / H
off, c = R - H / 2, S / 2
eye = Image.new("L", (S, S), 0)
ImageDraw.Draw(eye).ellipse([c - R, c + off - R, c + R, c + off + R], fill=255)
lower = Image.new("L", (S, S), 0)
ImageDraw.Draw(lower).ellipse([c - R, c - off - R, c + R, c - off + R], fill=255)
eye = Image.composite(eye, Image.new("L", (S, S), 0), lower)
img.paste((255, 255, 255, 255), (0, 0), eye)
d.ellipse([c - 165, c - 165, c + 165, c + 165], fill=(32, 48, 72))
d.ellipse([c - 60, c - 60, c + 60, c + 60], fill=(255, 255, 255))
lid = Image.new("L", (S, S), 0)                # the lid half-lowered: attention that may be acted
ImageDraw.Draw(lid).rectangle([0, 0, S, c - 40], fill=255)
lid = Image.composite(lid, Image.new("L", (S, S), 0), eye)
img.paste((240, 186, 150, 255), (0, 0), lid)
d.line([c - W / 2 + 80, c - 40, c + W / 2 - 80, c - 40], fill=(32, 48, 72), width=26)
out = Path(__file__).parent
img.save(out / "deceptive_gestures.png")
img.save(out / "deceptive_gestures.ico", sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
print("saved", out / "deceptive_gestures.ico")
