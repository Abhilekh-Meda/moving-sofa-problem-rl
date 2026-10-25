"""Animate a training run from its saved rotation paths (runs/<run>/paths/update_XXXX.npz).

learning.mp4  the deterministic policy's sofa after every update, in the sofa's frame,
              against Gerver's sofa at the same N, next to the eval-area curve
motion.mp4    the best update's sofa moving through the fixed L-shaped corridor
              (slide in, turn, slide out), beside Gerver's sofa doing the same
carving.mp4   a sequence of attempts (the policy after geometrically spaced updates, up to
              the best one): each carries the full starting block through the corridor
              and the walls shave it down to that attempt's sofa
discovery.mp4 carving.mp4 without Gerver's sofa in the sofa panel (its area stays on the
              curve); its shape is revealed only at the end, traced over the final sofa
every.mp4     discovery.mp4 with every update up to the best one as an attempt, each a quick pass

    python render.py runs/big/entropy_coef=0.001_seed1
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FFMpegWriter
from shapely import affinity

from sofa_env import SofaEnv, build_l_corridor, gerver_path, hammersley_path, replay
from sofa_env.reference import GERVER_AREA

LEARNED_COLOR = "#2f6b8f"
GERVER_COLOR = "#c0392b"
CORRIDOR_COLOR = "#e4e7eb"
WALL_COLOR = "#4a5560"
SHAVING_COLOR = "#e08a2c"
# certified area in the closing caption: half-plane directions and substeps per decision
# (6400 cut angles at N = 200; about a minute)
CERT_DIRS, CERT_SUBSTEPS = 64, 32


def _exterior(shape) -> np.ndarray:
    return np.asarray(shape.exterior.coords)


def _world_pose(shape, theta: float, corner: np.ndarray, shift=(0.0, 0.0)):
    """Sofa-frame shape -> world frame. The hallway in the sofa frame is c + R(theta) L,
    so a sofa point p sits at world point R(-theta) (p - c)."""
    moved = affinity.translate(shape, -corner[0], -corner[1])
    moved = affinity.rotate(moved, -theta, origin=(0, 0), use_radians=True)
    return affinity.translate(moved, *shift)


def _motion_poses(corners: np.ndarray, slide: float, slide_frames: int, turn_frames: int):
    """(theta, corner, shift) for every frame: slide in along the horizontal arm, turn, slide out down."""
    n_steps = len(corners) - 1
    poses = [(0.0, corners[0], (-d, 0.0)) for d in np.linspace(slide, 0.0, slide_frames)]
    for k in np.linspace(0, n_steps, turn_frames).round().astype(int):
        poses.append((k * (np.pi / 2) / n_steps, corners[k], (0.0, 0.0)))
    poses += [(np.pi / 2, corners[-1], (0.0, -d)) for d in np.linspace(0.0, slide, slide_frames)]
    return poses


def render_learning(run_dir: Path, out: Path, fps: int) -> None:
    files = sorted((run_dir / "paths").glob("update_*.npz"))
    updates = [np.load(f) for f in files]
    n_steps = int(updates[0]["n_steps"])
    gerver = replay(gerver_path(n_steps)).shape
    gerver_xy = _exterior(gerver)
    areas = np.array([float(z["area"]) for z in updates])

    fig, (ax_sofa, ax_curve) = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={"width_ratios": [1.6, 1]})
    writer = FFMpegWriter(fps=fps, bitrate=4000)
    with writer.saving(fig, str(out), dpi=120):
        for i, z in enumerate(updates + [updates[-1]] * (2 * fps)):
            j = min(i, len(updates) - 1)
            ax_sofa.clear()
            poly = z["polygon"]
            if len(poly) > 2:
                ax_sofa.fill(poly[:, 0], poly[:, 1], color=LEARNED_COLOR, alpha=0.35, lw=0)
                ax_sofa.plot(poly[:, 0], poly[:, 1], color=LEARNED_COLOR, lw=1.5, label="learned")
            ax_sofa.plot(gerver_xy[:, 0], gerver_xy[:, 1], color=GERVER_COLOR, lw=1.2, ls="--", label="Gerver")
            ax_sofa.set_xlim(-2.5, 1.2)
            ax_sofa.set_ylim(-0.15, 1.1)
            ax_sofa.set_aspect("equal")
            ax_sofa.legend(loc="upper left", fontsize=9, frameon=False)
            ax_sofa.set_title(f"update {int(z['update']) + 1}   area {areas[j]:.4f}   "
                              f"Gerver at N={n_steps}: {gerver.area:.4f}", fontsize=11)

            ax_curve.clear()
            ax_curve.plot(np.arange(1, j + 2), areas[: j + 1], color=LEARNED_COLOR, lw=1.5)
            ax_curve.plot(j + 1, areas[j], "o", color=LEARNED_COLOR)
            ax_curve.axhline(gerver.area, color=GERVER_COLOR, ls="--", lw=1)
            ax_curve.axhline(np.pi / 2, color=WALL_COLOR, ls=":", lw=1)
            ax_curve.text(len(areas), gerver.area, "Gerver", color=GERVER_COLOR, ha="right", va="bottom", fontsize=9)
            ax_curve.text(len(areas), np.pi / 2, "semicircle", color=WALL_COLOR, ha="right", va="bottom", fontsize=9)
            ax_curve.set_xlim(0, len(areas) + 1)
            ax_curve.set_ylim(min(1.4, areas.min() - 0.05), gerver.area + 0.05)
            ax_curve.set_xlabel("PPO update")
            ax_curve.set_ylabel("sofa area (deterministic policy)")
            fig.tight_layout()
            writer.grab_frame()
    plt.close(fig)


def render_motion(run_dir: Path, out: Path, fps: int) -> None:
    best = torch.load(run_dir / "best.pt", weights_only=False)
    z = np.load(run_dir / "paths" / f"update_{best['update']:04d}.npz")
    corners = z["corners"]
    n_steps = len(corners) - 1
    learned = replay(corners).shape
    g_corners = gerver_path(n_steps)
    gerver = replay(g_corners).shape

    corridor = _exterior(build_l_corridor(width=1.0, arm_length=6.0))
    slide = 3.6
    poses = _motion_poses(corners, slide, slide_frames=fps * 2, turn_frames=fps * 6)
    g_poses = _motion_poses(g_corners, slide, slide_frames=fps * 2, turn_frames=fps * 6)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.6))
    writer = FFMpegWriter(fps=fps, bitrate=4000)
    sofas = [(learned, poses, LEARNED_COLOR, f"learned, area {learned.area:.4f}"),
             (gerver, g_poses, GERVER_COLOR, f"Gerver, area {gerver.area:.4f}")]
    with writer.saving(fig, str(out), dpi=120):
        for f in range(len(poses) + fps):
            for ax, (shape, ps, color, label) in zip(axes, sofas):
                ax.clear()
                ax.fill(corridor[:, 0], corridor[:, 1], color=CORRIDOR_COLOR, lw=0)
                ax.plot(corridor[:, 0], corridor[:, 1], color=WALL_COLOR, lw=1.5)
                theta, corner, shift = ps[min(f, len(ps) - 1)]
                xy = _exterior(_world_pose(shape, theta, corner, shift))
                ax.fill(xy[:, 0], xy[:, 1], color=color, alpha=0.8, lw=0)
                ax.set_xlim(-5.5, 1.5)
                ax.set_ylim(-5.5, 1.5)
                ax.set_aspect("equal")
                ax.axis("off")
                ax.set_title(f"{label}   rotation {np.degrees(theta):4.1f}°", fontsize=11)
            fig.tight_layout()
            writer.grab_frame()
    plt.close(fig)


def _carve(corners: np.ndarray, max_corner_speed: float) -> list:
    """The shapes S_0, ..., S_N along a rotation path (sofa frame); replay() keeps only S_N."""
    env = SofaEnv(n_steps=len(corners) - 1, max_corner_speed=max_corner_speed)
    env.reset()
    shapes = [env.shape]
    for action in np.diff(corners, axis=0) / env.max_step:
        env.step(np.clip(action, -1.0, 1.0))
        shapes.append(env.shape)
    return shapes


def _attempt_updates(n_updates: int, best: int, count: int) -> list:
    """Geometrically spaced updates (dense early, where the shape changes most), ending at the best one."""
    picks = np.unique(np.geomspace(1, best + 1, count).round().astype(int) - 1)
    return [int(u) for u in picks if u < n_updates]


def _fill(ax, shape, **kwargs) -> None:
    if not shape.is_empty:
        xy = _exterior(shape)
        ax.fill(xy[:, 0], xy[:, 1], **kwargs)


class _SlowAfter:
    """Wraps an FFMpegWriter running `speed` times faster than real time: once `seconds` of
    video have been written, each frame is written `speed` times, so the rest plays at 1x."""

    def __init__(self, writer, speed: int, seconds: float):
        self.writer, self.speed, self.after = writer, speed, round(seconds * writer.fps)
        self.written = 0

    def grab_frame(self):
        for _ in range(self.speed if self.written >= self.after else 1):
            self.writer.grab_frame()
            self.written += 1


def _reveal(fig, writer, learned, gerver, certified: float, fps: int) -> None:
    """Closing frames: the final learned sofa alone, then Gerver's sofa fading in beside it
    behind an equals sign."""
    fig.clf()
    grid = fig.add_gridspec(1, 3, width_ratios=[1, 0.15, 1], left=0.02, right=0.98, wspace=0.02)
    ax_learned, ax_eq, ax_gerver = (fig.add_subplot(grid[0, j]) for j in range(3))
    alone, fade, hold = fps * 2, fps, fps * 4
    for f in range(alone + fade + hold):
        shown = np.clip((f - alone + 1) / fade, 0, 1)
        for ax in (ax_learned, ax_eq, ax_gerver):
            ax.clear()
            ax.axis("off")
        sides = [(ax_learned, learned, LEARNED_COLOR, "learned sofa", f"area ≥ {certified:.4f}", 1.0),
                 (ax_gerver, gerver, GERVER_COLOR, "Gerver's sofa", f"area {GERVER_AREA:.4f}", shown)]
        for ax, shape, color, name, area, alpha in sides:
            xy = _exterior(shape)
            ax.fill(xy[:, 0], xy[:, 1], color=color, alpha=0.35 * alpha, lw=0)
            ax.plot(xy[:, 0], xy[:, 1], color=color, lw=2, alpha=alpha)
            ax.text(-0.6, 1.15, name, color=color, ha="center", fontsize=24, alpha=alpha)
            ax.text(-0.6, -0.12, area, color=WALL_COLOR, ha="center", va="top", fontsize=15, alpha=alpha)
            ax.set_xlim(-2.3, 1.1)
            ax.set_ylim(-0.4, 1.4)
            ax.set_aspect("equal")
        ax_eq.text(0.5, 0.5, "=", ha="center", va="center", fontsize=60, color=WALL_COLOR, alpha=shown,
                   transform=ax_eq.transAxes)
        writer.grab_frame()


def render_carving(run_dir: Path, out: Path, fps: int, attempts: int, reveal: bool = False,
                   every: bool = False, speed: float = 1.0, slow_after: Optional[float] = None) -> None:
    """Each attempt (the deterministic policy after one update) carries the full starting
    block through the fixed corridor; the walls shave it down to that attempt's sofa.
    Side panels: the same carving in the sofa's frame against the earlier attempts' sofas,
    and the area curve with Gerver's area marked. Without `reveal`, Gerver's sofa is also
    drawn in the sofa panel throughout, with update numbers in the titles. With it, the
    titles show only the area, the curve builds up as training goes (no preview of later
    updates), only the last two sofas stay drawn, and Gerver's sofa appears only at the
    end (_reveal). With `every`, every update up to the best one is an attempt, each a
    quick pass. `speed` plays the same frames faster (the video's frame rate is fps * speed);
    with `slow_after`, the video returns to 1x after that many seconds (integer `speed` only)."""
    cfg = json.loads((run_dir / "config.json").read_text())
    speed = cfg["env"]["max_corner_speed"]
    files = sorted((run_dir / "paths").glob("update_*.npz"))
    updates = [np.load(f) for f in files]
    areas = np.array([float(z["area"]) for z in updates])
    best = torch.load(run_dir / "best.pt", weights_only=False)["update"]
    picks = list(range(best + 1)) if every else _attempt_updates(len(updates), best, attempts)
    n_steps = int(updates[0]["n_steps"])
    gerver = replay(gerver_path(n_steps)).shape
    gerver_xy = _exterior(gerver)
    hammersley_area = replay(hammersley_path(n_steps)).shape.area

    corridor = _exterior(build_l_corridor(width=1.0, arm_length=12.0))
    slide_in, slide_out, trail = 3.0, 1.5, 5
    if every:
        slide_frames, turn_frames, hold_frames = 2, 8, 1
    else:
        slide_frames, turn_frames, hold_frames = fps // 2, fps * 2, fps // 2
    kept = 2 if reveal else None

    fig = plt.figure(figsize=(14, 7))
    grid = fig.add_gridspec(2, 2, width_ratios=[1, 1.25], height_ratios=[1, 1])
    ax_world, ax_sofa, ax_curve = fig.add_subplot(grid[:, 0]), fig.add_subplot(grid[0, 1]), fig.add_subplot(grid[1, 1])
    writer = FFMpegWriter(fps=round(fps * speed), bitrate=5000)
    frames_out = writer if slow_after is None else _SlowAfter(writer, int(speed), slow_after)
    finished = []
    # the area axis only zooms in: attempt u's floor is set by the lowest area from update u/2
    # to the end of the run, which can only rise as u grows; the limits ease toward each target
    spans = [max(gerver.area - areas[(u + 1) // 2:].min(), 0.002) for u in picks]
    targets = [(gerver.area - 1.15 * span, gerver.area + 0.15 * span) for span in spans]
    y_low, y_top = targets[0]
    with writer.saving(fig, str(out), dpi=110):
        for i, u in enumerate(picks):
            corners = updates[u]["corners"]
            shapes = _carve(corners, speed)
            ks = np.linspace(0, n_steps, turn_frames).round().astype(int)
            # (step index, theta, shift): slide in, turn, slide out, hold
            frames = [(0, 0.0, (-d, 0.0)) for d in np.linspace(slide_in, 0.0, slide_frames)]
            frames += [(k, k * (np.pi / 2) / n_steps, (0.0, 0.0)) for k in ks]
            frames += [(n_steps, np.pi / 2, (0.0, -d)) for d in np.linspace(0.0, slide_out, slide_frames)]
            frames += [frames[-1]] * hold_frames
            target_low, target_top = targets[i]
            shavings = []
            prev_k = 0
            for k, theta, shift in frames:
                corner = corners[k]
                # material removed since the previous frame, drawn at the current pose
                if k != prev_k:
                    shavings.append(shapes[prev_k].difference(shapes[k]))
                shavings = shavings[-trail:]
                prev_k = k

                ax_world.clear()
                ax_world.fill(corridor[:, 0], corridor[:, 1], color=CORRIDOR_COLOR, lw=0)
                ax_world.plot(corridor[:, 0], corridor[:, 1], color=WALL_COLOR, lw=1.5)
                for age, cut in enumerate(reversed(shavings)):
                    for piece in getattr(cut, "geoms", [cut]):
                        _fill(ax_world, _world_pose(piece, theta, corner, shift),
                              color=SHAVING_COLOR, alpha=0.7 * (1 - age / trail), lw=0)
                _fill(ax_world, _world_pose(shapes[k], theta, corner, shift), color=LEARNED_COLOR, alpha=0.85, lw=0)
                ax_world.set_xlim(-4.5, 1.5)
                ax_world.set_ylim(-4.5, 1.5)
                ax_world.set_aspect("equal")
                ax_world.axis("off")
                if not reveal:
                    ax_world.set_title(f"attempt {i + 1}/{len(picks)}: policy after update {u + 1}   "
                                       f"rotation {np.degrees(theta):4.1f}°", fontsize=11)

                ax_sofa.clear()
                for done in (finished[-kept:] if kept else finished):
                    if done.is_empty:
                        continue
                    xy = _exterior(done)
                    ax_sofa.plot(xy[:, 0], xy[:, 1], color=LEARNED_COLOR, lw=0.8, alpha=0.25)
                _fill(ax_sofa, shapes[k], color=LEARNED_COLOR, alpha=0.35, lw=0)
                if not shapes[k].is_empty:
                    xy = _exterior(shapes[k])
                    ax_sofa.plot(xy[:, 0], xy[:, 1], color=LEARNED_COLOR, lw=1.5)
                ax_sofa.set_xlim(-3.4, 1.2)
                ax_sofa.set_ylim(-0.15, 1.1)
                ax_sofa.set_aspect("equal")
                if reveal:
                    ax_sofa.set_title(f"area {shapes[k].area:.4f}", fontsize=13)
                else:
                    ax_sofa.plot(gerver_xy[:, 0], gerver_xy[:, 1], color=GERVER_COLOR, lw=1.2, ls="--", label="Gerver")
                    ax_sofa.legend(loc="upper left", fontsize=9, frameon=False)
                    ax_sofa.set_title(f"in the sofa's frame   area {shapes[k].area:.4f}   "
                                      f"Gerver at N={n_steps}: {gerver.area:.4f}", fontsize=11)

                ax_curve.clear()
                if not reveal:
                    ax_curve.plot(np.arange(1, len(areas) + 1), areas, color=LEARNED_COLOR, lw=1, alpha=0.25)
                ax_curve.plot(np.arange(1, u + 2), areas[: u + 1], color=LEARNED_COLOR, lw=1.5)
                ax_curve.plot([p + 1 for p in picks[: i + 1]], areas[picks[: i + 1]], "o", color=LEARNED_COLOR, ms=4)
                for level, name, color, ls, va in [(gerver.area, "Gerver", GERVER_COLOR, "--", "bottom"),
                                                   (hammersley_area, "Hammersley", WALL_COLOR, "-.", "top"),
                                                   (np.pi / 2, "semicircle", WALL_COLOR, ":", "bottom")]:
                    ax_curve.axhline(level, color=color, ls=ls, lw=1)
                    ax_curve.text(len(areas), level, name, color=color, ha="right", va=va, fontsize=9, clip_on=True)
                y_low += 0.12 * (target_low - y_low)
                y_top += 0.12 * (target_top - y_top)
                ax_curve.set_xlim(0, len(areas) + 1)
                ax_curve.set_ylim(y_low, y_top)
                ax_curve.set_xlabel("PPO update")
                ax_curve.set_ylabel("sofa area")
                if reveal:
                    ax_curve.set_xticklabels([])
                else:
                    gap = gerver.area - areas[u]
                    ax_curve.set_title(f"update {u + 1}: area {areas[u]:.4f}, "
                                       f"{abs(gap):.4f} {'below' if gap >= 0 else 'above'} Gerver", fontsize=11)
                fig.tight_layout()
                frames_out.grab_frame()
            finished.append(shapes[-1])
        if reveal:
            # the area at the training N overstates the continuous motion's area; the caption
            # uses the certified lower bound, and Gerver's outline is taken at a fine N
            certified = replay(updates[picks[-1]]["corners"], max_corner_speed=speed,
                               reward_area="certified", cert_dirs=CERT_DIRS, substeps=CERT_SUBSTEPS).area
            _reveal(fig, frames_out, finished[-1], replay(gerver_path(4000)).shape, certified, fps)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--out", type=Path, default=None, help="output directory (default: the run directory)")
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--attempts", type=int, default=12, help="attempts shown in carving.mp4")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="playback speed of carving, discovery and every (same frames, higher frame rate)")
    parser.add_argument("--slow-after", type=float, default=None,
                        help="seconds into the video after which it plays at 1x again (integer --speed)")
    parser.add_argument("--only", choices=["learning", "motion", "carving", "discovery", "every"], default=None)
    args = parser.parse_args()
    out = args.out or args.run_dir
    out.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((args.run_dir / "config.json").read_text())
    print(f"rendering {args.run_dir} (N = {cfg['env']['n_steps']})")
    renders = {
        "learning": lambda path: render_learning(args.run_dir, path, args.fps),
        "motion": lambda path: render_motion(args.run_dir, path, args.fps),
        "carving": lambda path: render_carving(args.run_dir, path, args.fps, args.attempts,
                                               speed=args.speed, slow_after=args.slow_after),
        "discovery": lambda path: render_carving(args.run_dir, path, args.fps, args.attempts, reveal=True,
                                                 speed=args.speed, slow_after=args.slow_after),
        "every": lambda path: render_carving(args.run_dir, path, args.fps, args.attempts, reveal=True, every=True,
                                                 speed=args.speed, slow_after=args.slow_after),
    }
    for name, render in renders.items():
        if args.only in (None, name):
            render(out / f"{name}.mp4")
            print(f"wrote {out / f'{name}.mp4'}")


if __name__ == "__main__":
    main()
