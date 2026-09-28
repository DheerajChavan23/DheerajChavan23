#!/usr/bin/env python3
"""Generate the animated GitHub profile banners with high-definition Atkinson dithering
and dynamic particle collapse.

Run from the repository root:
    python scripts/banner/generate.py
"""

from __future__ import annotations

import html
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageOps
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "assets/source/img.png"
ASSETS = ROOT / "assets"
LOGOS = Path(__file__).resolve().parent / "logos"
DATA = Path(__file__).resolve().parent / "data"

W, H = 1180, 610
LOOP_SECONDS = 14.2
INTRO_SECONDS = 3.2
TRAVELLER_COUNT = 900
SEED = 314159

ROWS = [
    ("Subject", "Dheeraj Chavan"),
    ("Role", "Data & AI Engineer"),
    ("Location", "Dublin, Ireland 🇮🇪"),
    ("Education", "MSc Business Analytics · UCD Smurfit"),
    ("Status", "Learning -> Building -> Shipping"),
    ("ToolChain", "Antigravity · Claude Code · Git"),
    ("Core.AI", "LLM Systems · Agentic Workflows · MCP"),
    ("Core.Data", "Pipelines · Star Schema · PostgreSQL"),
    ("Core.Analytics", "Predictive Modeling · Tableau · Pandas"),
    ("Core.Automation", "Playwright · Python · APIs"),
    ("Core.Infra", "Docker · GitHub Actions · AWS"),
    ("Grid.LinkedIn", "/in/dheeraj-chavan23"),
    ("Grid.GitHub", "DheerajChavan23"),
    ("Grid.LeetCode", "chavan_dheeraj"),
    ("Grid.Mail", "chavandheeraj165@gmail.com"),
]

THEMES = {
    "dark": {
        "bg": "#0A101F",
        "panel": "#0D1628",
        "panel2": "#101B30",
        "line": "#25344C",
        "muted": "#8291A8",
        "text": "#DDE7F5",
        "portrait": "#AA9BEF",
        "chrome": "#22D3EE",
        "accent": "#10B981",
        "shadow": "#02050B",
    },
    "light": {
        "bg": "#F6F8FA",
        "panel": "#FFFFFF",
        "panel2": "#EDF3F7",
        "line": "#CBD7E1",
        "muted": "#64748B",
        "text": "#172033",
        "portrait": "#4A3D7A",
        "chrome": "#0891B2",
        "accent": "#10B981",
        "shadow": "#AAB7C4",
    },
}


def make_logos() -> dict[str, Image.Image]:
    """Create clean 400px black-on-transparent silhouette sources."""
    LOGOS.mkdir(parents=True, exist_ok=True)
    size = 400
    logos: dict[str, Image.Image] = {}

    # 1. Agent / AI Network (AI & Agentic Systems)
    agent = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d_agent = ImageDraw.Draw(agent)
    center = (200, 200)
    d_agent.ellipse((145, 145, 255, 255), fill="black")
    num_nodes = 6
    orbit_r = 135
    node_r = 28
    for i in range(num_nodes):
        ang = i * math.tau / num_nodes - math.pi / 2
        nx = int(center[0] + orbit_r * math.cos(ang))
        ny = int(center[1] + orbit_r * math.sin(ang))
        d_agent.line([center, (nx, ny)], fill="black", width=16)
        d_agent.ellipse((nx - node_r, ny - node_r, nx + node_r, ny + node_r), fill="black")
        next_ang = (i + 1) * math.tau / num_nodes - math.pi / 2
        nnx = int(center[0] + orbit_r * math.cos(next_ang))
        nny = int(center[1] + orbit_r * math.sin(next_ang))
        d_agent.line([(nx, ny), (nnx, nny)], fill="black", width=10)
    logos["agent"] = agent

    # 2. </> mark built from broad, rounded strokes (Engineering & Automation)
    code = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(code)
    stroke = 42
    d.line([(154, 95), (66, 200), (154, 305)], fill="black", width=stroke, joint="curve")
    d.line([(246, 95), (334, 200), (246, 305)], fill="black", width=stroke, joint="curve")
    d.line([(225, 72), (174, 328)], fill="black", width=stroke)
    logos["code"] = code

    # 3. Quality Shield with Checkmark (SDET, Verification & Analytics)
    shield = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d_shield = ImageDraw.Draw(shield)
    shield_pts = [
        (200, 48),
        (322, 92),
        (322, 225),
        (200, 352),
        (78, 225),
        (78, 92),
    ]
    d_shield.polygon(shield_pts, fill="black")
    check_pts = [(140, 195), (180, 240), (265, 140)]
    d_shield.line(check_pts, fill=(0, 0, 0, 0), width=32, joint="curve")
    logos["shield"] = shield

    for name, image in logos.items():
        image.save(LOGOS / f"{name}.png", optimize=True)
    return logos


def atkinson(gray_arr: np.ndarray) -> np.ndarray:
    """High-fidelity Atkinson error diffusion; preserves crisp edges and facial details."""
    work = gray_arr.copy().astype(np.float32) / 255.0
    h, w = work.shape
    out = np.zeros_like(work, dtype=bool)
    for y in range(h):
        for x in range(w):
            old = work[y, x]
            new = 1.0 if old >= 0.5 else 0.0
            out[y, x] = bool(new)
            err = (old - new) / 8.0
            if x + 1 < w:
                work[y, x + 1] += err
            if x + 2 < w:
                work[y, x + 2] += err
            if y + 1 < h:
                if x - 1 >= 0:
                    work[y + 1, x - 1] += err
                work[y + 1, x] += err
                if x + 1 < w:
                    work[y + 1, x + 1] += err
            if y + 2 < h:
                work[y + 2, x] += err
    return out


def portrait_points(theme: str, rng: np.random.Generator) -> np.ndarray:
    """Return sampled x/y banner coordinates from a crystal-clear Atkinson dither grid."""
    source = Image.open(SOURCE).convert("RGBA")
    # Tight crop: head, smile, and shoulders centered prominently in the 300x340 frame
    crop = source.crop((275, 160, 875, 840)).resize((300, 340), Image.Resampling.LANCZOS)
    rgb = crop.convert("RGB")
    alpha = np.asarray(crop.getchannel("A"), dtype=np.float32) / 255.0
    gray = np.asarray(ImageOps.grayscale(rgb), dtype=np.float32)

    # Tone curve tailored for Dheeraj's facial structure (protecting eyes, beard, and smile)
    g_img = Image.fromarray(np.uint8(np.clip(gray, 0, 255)), "L")
    g_img = ImageEnhance.Contrast(g_img).enhance(1.8)
    g_arr = (np.asarray(g_img) / 255.0) ** 1.2 * 255.0
    g_img = Image.fromarray(np.uint8(g_arr), "L")
    g_sharp = g_img.filter(ImageFilter.UnsharpMask(radius=2, percent=220, threshold=3))

    bits = atkinson(np.asarray(g_sharp))
    if theme == "dark":
        active = bits & (alpha > 0.1)
    else:
        active = (~bits) & (alpha > 0.1)

    ys, xs = np.where(active)
    if len(xs) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    points = np.column_stack((74 + xs, 154 + ys)).astype(np.float32)
    return points


def sample_logo_points(
    image: Image.Image, rng: np.random.Generator, count: int
) -> np.ndarray:
    """Sample a silhouette into the portrait frame's visual coordinate space."""
    alpha = np.asarray(image.getchannel("A"))
    ys, xs = np.where(alpha > 127)
    chosen = rng.choice(len(xs), count, replace=len(xs) < count)
    # Logo occupies a centered 270x270 square inside VISUAL.MAP.
    return np.column_stack((89 + xs[chosen] * 0.675, 188 + ys[chosen] * 0.675)).astype(
        np.float32
    )


def transport(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Order target points by minimum-cost assignment from source points."""
    rows, cols = linear_sum_assignment(cdist(source, target, metric="sqeuclidean"))
    ordered = np.empty_like(target)
    ordered[rows] = target[cols]
    return ordered


def num(value: float) -> str:
    return f"{value:.1f}".rstrip("0").rstrip(".")


def point_path(points: np.ndarray) -> str:
    """Aggregate adjacent horizontal one-pixel dots into compact SVG path runs."""
    if not len(points):
        return ""
    integer = np.rint(points).astype(int)
    unique = sorted({(int(x), int(y)) for x, y in integer}, key=lambda p: (p[1], p[0]))
    chunks: list[str] = []
    i = 0
    while i < len(unique):
        x0, y = unique[i]
        x1 = x0
        i += 1
        while i < len(unique) and unique[i][1] == y and unique[i][0] <= x1 + 1:
            x1 = unique[i][0]
            i += 1
        chunks.append(f"M{x0} {y}h{x1 - x0 + 1}")
    return "".join(chunks)


def dotted_leader(x1: float, x2: float, y: float) -> str:
    if x2 <= x1:
        return ""
    return "".join(f"M{x} {num(y)}h1" for x in np.arange(x1, x2, 5.0))


def text_width(text: str, font_size: float) -> float:
    """Stable monospace width used both for textLength and leader placement."""
    return len(text) * font_size * 0.605


def animate_values(points: list[np.ndarray], index: int) -> str:
    return ";".join(f"{num(p[index, 0])} {num(p[index, 1])}" for p in points)


def render_svg(
    theme_name: str,
    portrait: np.ndarray,
    logo_points: dict[str, np.ndarray],
    rng: np.random.Generator,
) -> str:
    t = THEMES[theme_name]
    n = min(TRAVELLER_COUNT, len(portrait))
    source = portrait[rng.choice(len(portrait), n, replace=False)]
    agent = transport(source, logo_points["agent"][:n])
    code = transport(agent, logo_points["code"][:n])
    shield = transport(code, logo_points["shield"][:n])

    # 14.2 second continuous loop
    times = [0, 3.0, 4.3, 6.3, 7.6, 9.6, 10.9, 12.9, 14.2]
    key_times = ";".join(num(v / LOOP_SECONDS) for v in times)
    frames = [source, source, agent, agent, code, code, shield, shield, source]
    # Travellers: invisible during portrait hold, visible during all logo phases, invisible upon return
    opacity_values = "0;0;1;1;1;1;1;1;0"

    parts: list[str] = [
        '<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" '
        'aria-labelledby="title desc">',
        "<title id=\"title\">Dheeraj's live system profile</title>",
        '<desc id="desc">Animated terminal profile with a sharp dithered portrait and '
        "agent, code, and quality assurance silhouettes.</desc>",
        "<defs>",
        '<filter id="shadow" x="-20%" y="-20%" width="140%" height="150%">'
        f'<feDropShadow dx="0" dy="12" stdDeviation="16" flood-color="{t["shadow"]}" '
        'flood-opacity=".28"/></filter>',
        '<filter id="glow" x="-100%" y="-100%" width="300%" height="300%">'
        f'<feGaussianBlur stdDeviation="3" result="b"/><feFlood flood-color="{t["chrome"]}" '
        'flood-opacity=".35"/><feComposite in2="b" operator="in"/>'
        '<feMerge><feMergeNode/><feMergeNode in="SourceGraphic"/></feMerge></filter>',
        '<clipPath id="visualClip"><rect x="49" y="124" width="390" height="414" rx="3"/></clipPath>',
        "</defs>",
        f'<rect width="{W}" height="{H}" rx="18" fill="{t["bg"]}"/>',
        f'<rect x="13" y="13" width="1154" height="584" rx="13" fill="{t["panel"]}" '
        f'stroke="{t["line"]}" filter="url(#shadow)"/>',
        f'<path d="M13 62H1167" stroke="{t["line"]}"/>',
        '<circle cx="38" cy="38" r="6" fill="#FF5F57"/>'
        '<circle cx="59" cy="38" r="6" fill="#FEBC2E"/>'
        '<circle cx="80" cy="38" r="6" fill="#28C840"/>',
        f'<text x="590" y="43" text-anchor="middle" fill="{t["muted"]}" '
        'font-family="ui-monospace,SFMono-Regular,Consolas,monospace" font-size="13" '
        'letter-spacing=".4">profile.sh --live</text>',
        # Left visual frame.
        f'<rect x="35" y="88" width="418" height="472" rx="6" fill="{t["panel2"]}" '
        f'stroke="{t["line"]}"/>',
        f'<path d="M35 124H453" stroke="{t["line"]}"/>',
        f'<text x="49" y="111" fill="{t["chrome"]}" '
        'font-family="ui-monospace,SFMono-Regular,Consolas,monospace" font-size="13" '
        'font-weight="700" letter-spacing="1.2">VISUAL.MAP</text>',
        f'<text x="438" y="111" text-anchor="end" fill="{t["muted"]}" '
        'font-family="ui-monospace,SFMono-Regular,Consolas,monospace" font-size="11">300×340 / ATKINSON</text>',
        f'<path d="M49 141h12M49 141v12M439 141h-12M439 141v12M49 539h12M49 539v-12'
        f'M439 539h-12M439 539v-12" fill="none" stroke="{t["chrome"]}" opacity=".55"/>',
        '<g clip-path="url(#visualClip)" shape-rendering="crispEdges">',
    ]

    # Dynamic particle collapse:
    # 94 bands that hold rock-solid at t=0..3.0s, then at t=3.0s break apart and collapse
    # toward the center while fading completely to OPACITY 0!
    # During t=4.3s to 12.9s (all logo displays), background opacity is 0 (COMPLETELY INVISIBLE).
    # Then at t=12.9s..14.2s, they smoothly reform as the particles return.
    agent_centroid = agent.mean(axis=0)
    band_ids = rng.integers(0, 94, size=len(portrait))
    noise = rng.normal(0, 5, size=(94, 2))
    for band in range(94):
        pts = portrait[band_ids == band]
        if not len(pts):
            continue
        centroid = pts.mean(axis=0)
        # Shift towards the logo center with dispersion noise
        delta = (agent_centroid - centroid) * 0.28 + noise[band]
        d = point_path(pts)
        parts.append(
            f'<path d="{d}" fill="none" stroke="{t["portrait"]}" stroke-width="1">'
            f'<animateTransform attributeName="transform" type="translate" begin="0s" '
            f'dur="{LOOP_SECONDS}s" repeatCount="indefinite" calcMode="linear" '
            f'keyTimes="{key_times}" values="0 0;0 0;{num(delta[0])} {num(delta[1])};'
            f'{num(delta[0])} {num(delta[1])};0 0;0 0;0 0;0 0;0 0"/>'
            f'<animate attributeName="opacity" begin="0s" dur="{LOOP_SECONDS}s" '
            f'repeatCount="indefinite" keyTimes="{key_times}" '
            'values="1;1;0;0;0;0;0;0;1"/></path>'
        )

    # Optimal-transport travellers: morph between portrait and the 3 logos
    for i in range(n):
        positions = animate_values(frames, i)
        parts.append(
            f'<path d="M-.65-.65h1.3v1.3h-1.3z" fill="{t["portrait"]}">'
            f'<animateTransform attributeName="transform" type="translate" begin="0s" '
            f'dur="{LOOP_SECONDS}s" repeatCount="indefinite" calcMode="linear" '
            f'keyTimes="{key_times}" values="{positions}"/>'
            f'<animate attributeName="opacity" begin="0s" dur="{LOOP_SECONDS}s" '
            f'repeatCount="indefinite" calcMode="linear" keyTimes="{key_times}" '
            f'values="{opacity_values}"/></path>'
        )

    parts.extend(
        [
            "</g>",
            # Telemetry footer
            f'<text x="58" y="551" fill="{t["muted"]}" '
            'font-family="ui-monospace,SFMono-Regular,Consolas,monospace" font-size="10">'
            f'PTS {len(portrait):05d} · HD/ATKINSON</text>',
            # Right information panel.
            f'<rect x="474" y="88" width="672" height="472" rx="6" fill="{t["panel2"]}" '
            f'stroke="{t["line"]}"/>',
            f'<path d="M474 124H1146" stroke="{t["line"]}"/>',
            f'<text x="490" y="111" fill="{t["chrome"]}" '
            'font-family="ui-monospace,SFMono-Regular,Consolas,monospace" font-size="13" '
            'font-weight="700" letter-spacing="1.2">SYSTEM.INFO</text>',
            # LIVE badge and handle pill.
            '<g filter="url(#glow)"><circle cx="915" cy="106" r="4" fill="#FF4D5A">'
            '<animate attributeName="opacity" values="1;.3;1" dur="1.6s" repeatCount="indefinite"/>'
            '</circle></g>',
            '<text x="927" y="111" fill="#FF4D5A" '
            'font-family="ui-monospace,SFMono-Regular,Consolas,monospace" font-size="12" '
            'font-weight="700">LIVE</text>',
            f'<rect x="960" y="94" width="168" height="24" rx="12" fill="{t["chrome"]}" opacity=".16" '
            f'stroke="{t["chrome"]}"/>',
            f'<text x="1044" y="111" text-anchor="middle" fill="{t["chrome"]}" '
            'font-family="ui-monospace,SFMono-Regular,Consolas,monospace" font-size="13" '
            'font-weight="700">@DheerajChavan23</text>',
        ]
    )

    value_right = 1127.0
    row_y = 153.0
    for label, value in ROWS:
        value_len = text_width(value, 14)
        label_len = text_width(label, 14)
        leader_start = 491 + label_len + 12
        leader_end = value_right - value_len - 12
        parts.extend(
            [
                f'<text x="491" y="{num(row_y)}" fill="{t["muted"]}" '
                'font-family="ui-monospace,SFMono-Regular,Consolas,monospace" font-size="14">'
                f"{html.escape(label)}</text>",
                f'<path d="{dotted_leader(leader_start, leader_end, row_y - 4)}" '
                f'fill="none" stroke="{t["line"]}" stroke-width="1" shape-rendering="crispEdges"/>',
                f'<text x="{num(value_right)}" y="{num(row_y)}" text-anchor="end" '
                f'fill="{t["text"]}" font-family="ui-monospace,SFMono-Regular,Consolas,monospace" '
                f'font-size="14" textLength="{num(value_len)}" lengthAdjust="spacingAndGlyphs">'
                f"{html.escape(value)}</text>",
            ]
        )
        row_y += 23

    parts.extend(
        [
            f'<path d="M490 530H1130" stroke="{t["line"]}"/>',
            f'<text x="491" y="548" fill="{t["accent"]}" '
            'font-family="ui-monospace,SFMono-Regular,Consolas,monospace" font-size="11">'
            "● ALL SYSTEMS NOMINAL</text>",
            f'<text x="1128" y="548" text-anchor="end" fill="{t["muted"]}" '
            'font-family="ui-monospace,SFMono-Regular,Consolas,monospace" font-size="11">'
            "UTC+1 · DUBLIN NODE</text>",
            "</svg>",
        ]
    )
    return "".join(parts)


def main() -> None:
    if not SOURCE.exists():
        raise SystemExit(f"Missing source portrait: {SOURCE}")
    ASSETS.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    logos = make_logos()

    portraits: dict[str, np.ndarray] = {}
    for index, theme in enumerate(THEMES):
        rng = np.random.default_rng(SEED + index)
        points = portrait_points(theme, rng)
        portraits[theme] = points
        np.save(DATA / f"portrait-{theme}.npy", points)

    for index, theme in enumerate(THEMES):
        rng = np.random.default_rng(SEED + 100 + index)
        sampled = {
            name: sample_logo_points(image, rng, TRAVELLER_COUNT)
            for name, image in logos.items()
        }
        for name, points in sampled.items():
            np.save(DATA / f"{name}-{theme}.npy", points)
        svg = render_svg(theme, portraits[theme], sampled, rng)
        output = ASSETS / f"banner-{theme}.v9.svg"
        output.write_text(svg, encoding="utf-8")
        byte_size = output.stat().st_size
        print(
            f"{output.relative_to(ROOT)}: {byte_size:,} bytes "
            f"({byte_size / 1024:.1f} KiB), {len(portraits[theme]):,} portrait dots, "
            f"{TRAVELLER_COUNT} travellers"
        )

    for name in logos:
        output = LOGOS / f"{name}.png"
        print(f"{output.relative_to(ROOT)}: {output.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
