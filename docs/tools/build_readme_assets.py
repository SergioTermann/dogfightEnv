"""Rebuild README artwork with Pillow (python -m pip install pillow).

Run from any directory: python docs/tools/build_readme_assets.py
Screenshots are existing repository assets; no generated gameplay is used.
SVG diagrams use native text and remain editable and accessible.
"""

from html import escape
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "images" / "readme"
INK, MUTED, LINE = "#20292D", "#59696F", "#D9E1E4"
TEAL, BLUE, RUST = "#007E75", "#2361AC", "#B45C37"


def font(size, bold=False):
    candidates = (
        "C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    )
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    raise RuntimeError("Install Segoe UI or DejaVu Sans to rebuild the raster assets.")


def cover():
    source = ROOT / "dogfight_sandbox_hg2/screenshots/screenshot_4.png"
    scene = Image.open(source).convert("RGB").crop((0, 80, 1080, 600))
    canvas = Image.new("RGB", (1600, 890), INK)
    canvas.paste(scene.resize((1600, 770), Image.Resampling.LANCZOS), (0, 0))
    d = ImageDraw.Draw(canvas)
    d.rectangle((64, 46, 101, 51), fill=INK)
    d.text((115, 30), "AIR COMBAT / RESEARCH ENVIRONMENT", fill=INK, font=font(22, True))
    d.text((58, 71), "dogfightEnv", fill=INK, font=font(100, True))
    d.text((65, 194), "Flight dynamics. Learning agents. Mission command.", fill=INK, font=font(29))
    for x, value, label in [(64, "6-DOF", "JSBSim flight dynamics"),
                            (570, "PPO / SAC / Rainbow", "PyTorch training suite"),
                            (1130, "Rule + LLM", "External mission commander")]:
        d.text((x, 789), value, fill="#FFFFFF", font=font(27, True))
        d.text((x, 836), label, fill="#B9CACD", font=font(21))
    for x in (530, 1090):
        d.line((x, 800, x, 856), fill="#516065", width=1)
    canvas.save(OUT / "cover.jpg", quality=94, subsampling=0, optimize=True)


class Diagram:
    def __init__(self, height, title, description):
        self.parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="1280" height="{height}" '
            f'viewBox="0 0 1280 {height}" role="img" aria-labelledby="title desc">',
            f'<title id="title">{escape(title)}</title><desc id="desc">{escape(description)}</desc>',
            '<defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" '
            'orient="auto-start-reverse"><path d="M0 0L8 4L0 8Z" fill="#708187"/></marker></defs>',
            f'<rect width="1280" height="{height}" fill="#FFFFFF"/>',
            '<g font-family="Segoe UI, Arial, sans-serif">',
        ]

    def text(self, x, y, value, size=22, color=INK, bold=False):
        self.parts.append(f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" '
                          f'font-weight="{700 if bold else 400}">{escape(value)}</text>')

    def rect(self, x, y, w, h, fill="white", stroke=LINE):
        self.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" '
                          f'rx="4" fill="{fill}" stroke="{stroke}"/>')

    def path(self, points, color=LINE, arrow=False, dashed=False):
        self.parts.append(f'<path d="{points}" fill="none" stroke="{color}" stroke-width="2"'
                          + (' marker-end="url(#arrow)"' if arrow else '')
                          + (' stroke-dasharray="7 6"' if dashed else '') + '/>')

    def heading(self, number, title, subtitle):
        self.text(48, 45, number, 18, TEAL, True)
        self.text(48, 94, title, 36, bold=True)
        self.text(48, 132, subtitle, 21, MUTED)
        self.path("M48 161H1232")

    def save(self, name):
        (OUT / name).write_text("\n".join(self.parts + ["</g></svg>"]), encoding="utf-8")


def architecture():
    d = Diagram(940, "dogfightEnv runtime architecture",
                "One external TCP client, either RL or commander, connects to the sandbox. "
                "Keyboard and gamepad inputs are local. JSBSim updates flight dynamics; "
                "Harfang renders the world. Logs and checkpoints are written by their owning clients.")
    d.heading("01 / SYSTEM ARCHITECTURE", "A shared simulation. Two external control modes.",
              "Process boundaries, control ownership and experiment outputs.")
    d.text(48, 208, "EXTERNAL CLIENT / SELECT ONE", 18, MUTED, True)
    d.text(624, 208, "SANDBOX PROCESS", 18, MUTED, True)
    d.rect(48, 230, 432, 170, "#EFF8F6", "#B8DAD4")
    d.text(72, 266, "01A / LEARNING", 17, TEAL, True)
    d.text(72, 304, "RL environment + policy", 26, bold=True)
    d.text(72, 340, "Gym adapter / PPO / SAC / Rainbow", 21, MUTED)
    d.text(72, 375, "Actions in; observations and rewards out", 19, TEAL)
    d.rect(48, 424, 432, 170, "#FCF4EE", "#E5CBBC")
    d.text(72, 460, "01B / MISSION COMMAND", 17, RUST, True)
    d.text(72, 498, "Rule or LLM commander", 26, bold=True)
    d.text(72, 534, "engage / patrol / retreat / hold", 21, MUTED)
    d.text(72, 569, "Poll state; apply changed assignments", 20, RUST)
    d.path("M480 316H544V510H480", "#708187")
    d.path("M544 412H624", "#708187", True)
    d.text(551, 392, "TCP", 17, MUTED, True)
    d.rect(624, 230, 608, 96, "#EFF5FC", "#C4D5E9")
    d.text(648, 267, "NETWORK INTERFACE", 17, BLUE, True)
    d.text(648, 303, "JSON over TCP / host:50888 / one client", 24, bold=True)
    d.path("M920 326V354", "#708187", True)
    d.rect(624, 354, 608, 116, "#F7F9FA")
    d.text(648, 393, "World state + aircraft control", 27, bold=True)
    d.text(648, 431, "Missions / targets / built-in IA / autopilot", 22, MUTED)
    d.path("M920 470V504", "#708187")
    d.path("M774 531V504H1077V531", "#708187")
    d.rect(624, 531, 292, 136, "#EFF8F6", "#B8DAD4")
    d.text(648, 570, "JSBSim", 28, TEAL, True)
    d.text(648, 608, "F-16 flight dynamics", 22)
    d.text(648, 643, "Fixed FDM step: 1/60 s", 20, MUTED)
    d.rect(940, 531, 292, 136, "#EFF5FC", "#C4D5E9")
    d.text(964, 570, "Harfang3D", 28, BLUE, True)
    d.text(964, 608, "3D world + cameras", 22)
    d.text(964, 643, "HUD + predicted paths", 20, MUTED)
    d.text(648, 704, "Missiles retain the sandbox guidance / physics model.", 19, MUTED)
    d.rect(48, 650, 432, 100, "#F7F9FA")
    d.text(72, 689, "Keyboard / Xbox gamepad", 26, bold=True)
    d.text(72, 726, "Local input inside the sandbox process", 20, MUTED)
    d.path("M480 697H581V451H624", "#708187", True, True)
    d.path("M48 794H1232")
    d.text(48, 834, "EXPERIMENT OUTPUTS", 17, MUTED, True)
    d.text(48, 876, "Training", 23, TEAL, True)
    d.text(48, 910, "Checkpoints + metrics", 21, MUTED)
    d.text(480, 876, "Commander", 23, RUST, True)
    d.text(480, 910, "Decision log (JSONL)", 21, MUTED)
    d.text(900, 876, "Environment", 23, BLUE, True)
    d.text(900, 910, "Tacview ACMI (1v1)", 21, MUTED)
    d.save("architecture.svg")


def workflow():
    d = Diagram(650, "Training and commander execution modes",
                "Training controls scene stepping. The commander observes and issues tasks "
                "while the sandbox runs freely. These modes use separate sessions, not concurrent clients.")
    d.heading("02 / EXECUTION MODEL", "Control the step. Or command the mission.",
              "Two workflows with different simulation-clock ownership.")
    for top, color, label, note, items in [
        (198, TEAL, "A / RL TRAINING", "Client-driven scene updates in renderless mode",
         [("Observe", "State + reward"), ("Act", "Policy output"),
          ("Step", "update_scene"), ("Learn", "Update + save")]),
        (404, RUST, "B / COMMANDER", "Free-running sandbox; commander does not drive scene updates",
         [("Observe", "Battle snapshot"), ("Decide", "Rule or LLM"),
          ("Validate", "Names + targets"), ("Apply", "Changed tasks")]),
    ]:
        d.text(48, top, label, 19, color, True)
        d.text(48, top + 34, note, 21, MUTED)
        for i, (title, subtitle) in enumerate(items):
            x, y = 48 + i * 304, top + 57
            d.rect(x, y, 272, 100, "#F7F9FA")
            d.text(x + 20, y + 39, f"0{i + 1}  {title}", 25, color, True)
            d.text(x + 20, y + 75, subtitle, 21, MUTED)
            if i < 3:
                d.path(f"M{x + 272} {y + 50}h29", "#708187", True)
    d.path("M48 596H1232")
    d.text(48, 630, "ONE TCP CLIENT PER SANDBOX / RUN THESE WORKFLOWS IN SEPARATE SESSIONS", 18, MUTED, True)
    d.save("execution.svg")


def gallery():
    canvas = Image.new("RGB", (1600, 1065), "#FFFFFF")
    d = ImageDraw.Draw(canvas)
    d.text((48, 25), "SIMULATION / IN-ENGINE CAPTURES", fill=TEAL, font=font(21, True))
    d.text((48, 65), "Observe the engagement.", fill=INK, font=font(48, True))
    images = [
        ("screenshot_commander_engage.png", "01 / MISSION ASSIGNMENTS", "Aircraft tasks in the 3D view"),
        ("screenshot_prediction.png", "02 / FLIGHT PREDICTION", "Kinematic paths with time markers"),
        ("view_left.png", "03 / EXTERNAL CAMERA", "Aircraft and surrounding airspace"),
        ("view_cockpit.png", "04 / COCKPIT CAMERA", "Pilot perspective and instrumentation"),
    ]
    for i, (file, title, caption) in enumerate(images):
        x, y = 48 + (i % 2) * 776, 147 + (i // 2) * 451
        source = Image.open(ROOT / "docs/images" / file).convert("RGB")
        # Remove only the desktop/window chrome; retain the full simulation viewport.
        source = source.crop((8, 31, 1928, 1111))
        canvas.paste(source.resize((728, 410), Image.Resampling.LANCZOS), (x, y))
        # Captions sit on an opaque footer without covering simulation details.
        d.rectangle((x, y + 338, x + 728, y + 410), fill=INK)
        d.text((x + 18, y + 347), title, fill="white", font=font(20, True))
        d.text((x + 18, y + 378), caption, fill="#B9CACD", font=font(19))
    canvas.save(OUT / "simulation.jpg", quality=92, subsampling=0, optimize=True)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    cover()
    architecture()
    workflow()
    gallery()
    print(f"Built 4 assets in {OUT}")
