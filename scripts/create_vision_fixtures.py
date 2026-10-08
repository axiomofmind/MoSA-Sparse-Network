"""Create deterministic local images for Gemma admission tests."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _save_screenshot(root: Path) -> None:
    image = Image.new("RGB", (640, 360), "#10151f")
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 640, 42), fill="#263248")
    draw.text((18, 11), "Sparse Build Dashboard", font=_font(18), fill="white")
    draw.rectangle((18, 64, 180, 330), fill="#1c2637")
    draw.text((34, 85), "Jobs", font=_font(20), fill="#b9c7dd")
    draw.rectangle((205, 64, 620, 150), fill="#742f38")
    draw.text((225, 80), "BUILD FAILED", font=_font(26), fill="white")
    draw.text((225, 116), "ModuleNotFoundError: sparse_utils", font=_font(18), fill="white")
    draw.rectangle((205, 174, 620, 330), fill="#182234")
    draw.text((225, 195), "3 tests passed", font=_font(19), fill="#6de29a")
    draw.text((225, 235), "1 import error", font=_font(19), fill="#ff8b8b")
    image.save(root / "screenshot.png")


def _save_document(root: Path) -> None:
    image = Image.new("RGB", (500, 650), "white")
    draw = ImageDraw.Draw(image)
    draw.text((45, 45), "INVOICE", font=_font(34), fill="#172033")
    draw.text((45, 105), "Invoice: INV-204", font=_font(22), fill="black")
    draw.text((45, 145), "Customer: Ada Example", font=_font(22), fill="black")
    draw.line((45, 205, 455, 205), fill="#68758b", width=2)
    draw.text((45, 235), "Local inference service", font=_font(20), fill="black")
    draw.text((370, 235), "$42.50", font=_font(20), fill="black")
    draw.line((45, 300, 455, 300), fill="#68758b", width=2)
    draw.text((285, 330), "TOTAL: $42.50", font=_font(24), fill="#172033")
    image.save(root / "document.png")


def _save_chart(root: Path) -> None:
    image = Image.new("RGB", (640, 420), "white")
    draw = ImageDraw.Draw(image)
    draw.text((170, 20), "Verified Tasks by Agent", font=_font(26), fill="#172033")
    draw.line((85, 350, 585, 350), fill="black", width=3)
    draw.line((85, 80, 85, 350), fill="black", width=3)
    bars = [
        (135, 20, "BLUE", "#3478d4"),
        (285, 55, "GREEN", "#32a66a"),
        (435, 35, "ORANGE", "#e58a2b"),
    ]
    for x, value, label, color in bars:
        top = 350 - value * 4
        draw.rectangle((x, top, x + 90, 350), fill=color)
        draw.text((x + 25, top - 30), str(value), font=_font(20), fill="black")
        draw.text((x + 8, 365), label, font=_font(17), fill="black")
    image.save(root / "chart.png")


def _save_general(root: Path) -> None:
    image = Image.new("RGB", (480, 320), "#f1f5fa")
    draw = ImageDraw.Draw(image)
    draw.ellipse((55, 70, 225, 240), fill="#2878d0")
    draw.rectangle((280, 75, 445, 240), fill="#d33f49")
    draw.text((92, 255), "BLUE CIRCLE", font=_font(18), fill="black")
    draw.text((302, 255), "RED SQUARE", font=_font(18), fill="black")
    image.save(root / "general.png")


def _save_diagram(root: Path) -> None:
    image = Image.new("RGB", (720, 300), "white")
    draw = ImageDraw.Draw(image)
    boxes = [
        ((35, 95, 190, 205), "REQUEST", "#d8e8ff"),
        ((280, 95, 435, 205), "ROUTER", "#fff0c7"),
        ((525, 95, 680, 205), "GEMMA", "#daf4df"),
    ]
    for bounds, label, color in boxes:
        draw.rounded_rectangle(bounds, radius=12, fill=color, outline="#263248", width=3)
        draw.text((bounds[0] + 25, 135), label, font=_font(24), fill="#172033")
    for start, end in [((190, 150), (280, 150)), ((435, 150), (525, 150))]:
        draw.line((*start, *end), fill="#263248", width=5)
        draw.polygon(
            [(end[0], end[1]), (end[0] - 18, end[1] - 11), (end[0] - 18, end[1] + 11)],
            fill="#263248",
        )
    image.save(root / "diagram.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    _save_screenshot(root)
    _save_document(root)
    _save_chart(root)
    _save_general(root)
    _save_diagram(root)
    print(root)


if __name__ == "__main__":
    main()
