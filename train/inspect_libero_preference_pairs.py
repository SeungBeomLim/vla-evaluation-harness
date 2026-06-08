"""Create visual inspection sheets for LIBERO chunk preference pairs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", required=True, help="preference_pairs.jsonl or preference_pairs_with_idm.jsonl.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--idm-actions", default=None, help="Optional idm_actions.npz.")
    parser.add_argument("--num-samples", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--selection", choices=["random", "offset13", "high-score"], default="random")
    parser.add_argument("--view", action="append", default=None, help="Image view to render. Repeatable.")
    parser.add_argument("--context", type=int, default=2, help="Frames around target_step to include.")
    parser.add_argument("--image-width", type=int, default=160)
    return parser.parse_args()


def _load_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _select_rows(rows: list[dict[str, Any]], *, mode: str, n: int, seed: int) -> list[tuple[int, dict[str, Any]]]:
    indexed = list(enumerate(rows))
    if mode == "offset13":
        indexed = [(i, r) for i, r in indexed if int(r.get("action_offset", -1)) == 13]
    elif mode == "high-score":
        indexed.sort(key=lambda x: float(x[1].get("selection_debug", {}).get("target_score", 0.0)), reverse=True)
        return indexed[:n]
    rng = random.Random(seed)
    rng.shuffle(indexed)
    return indexed[:n]


def _trajectory_path(ref: dict[str, Any]) -> Path:
    return Path(str(ref["root"])) / Path(str(ref["trajectory"]))


def _image_at(ref: dict[str, Any], view: str, step: int) -> Image.Image:
    key = f"image_{view}"
    with np.load(_trajectory_path(ref)) as traj:
        if key not in traj.files:
            raise KeyError(f"{key} not found in {_trajectory_path(ref)}")
        step = max(0, min(int(step), len(traj[key]) - 1))
        arr = traj[key][step]
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def _resize(img: Image.Image, width: int) -> Image.Image:
    scale = width / img.width
    height = max(1, int(round(img.height * scale)))
    return img.resize((width, height), Image.Resampling.BILINEAR)


def _font(size: int = 14) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except Exception:
        return ImageFont.load_default()


def _captioned(img: Image.Image, caption: str, *, width: int) -> Image.Image:
    img = _resize(img, width)
    pad = 6
    text_h = 38
    out = Image.new("RGB", (img.width, img.height + text_h), "white")
    out.paste(img, (0, text_h))
    draw = ImageDraw.Draw(out)
    draw.multiline_text((pad, pad), caption, fill="black", font=_font(12), spacing=2)
    return out


def _concat_grid(images: list[Image.Image], *, cols: int, gap: int = 8) -> Image.Image:
    if not images:
        raise ValueError("No images to render")
    rows = (len(images) + cols - 1) // cols
    cell_w = max(img.width for img in images)
    cell_h = max(img.height for img in images)
    out = Image.new("RGB", (cols * cell_w + (cols - 1) * gap, rows * cell_h + (rows - 1) * gap), "white")
    for idx, img in enumerate(images):
        r, c = divmod(idx, cols)
        x = c * (cell_w + gap)
        y = r * (cell_h + gap)
        out.paste(img, (x, y))
    return out


def _idm_stats(row: dict[str, Any], idm_actions: np.ndarray | None) -> dict[str, float] | None:
    if idm_actions is None or "idm" not in row:
        return None
    idx = int(row["idm"]["index"])
    action = idm_actions[idx]
    return {
        "idm_l2": float(np.linalg.norm(action)),
        "idm_pos_l2": float(np.linalg.norm(action[:, 0:3])),
        "idm_rot_l2": float(np.linalg.norm(action[:, 3:6])),
        "idm_grip_l2": float(np.linalg.norm(action[:, 6:])),
    }


def _make_sheet(
    *,
    row_idx: int,
    row: dict[str, Any],
    view: str,
    output_dir: Path,
    context: int,
    image_width: int,
    idm_actions: np.ndarray | None,
) -> dict[str, Any]:
    c = int(row["policy_obs_step"])
    t = int(row["target_step"])
    offset = int(row["action_offset"])
    failure_policy = row["vla_input"]
    failure_target = row["negative"]
    success_target = row["positive"]

    cells: list[Image.Image] = []
    cells.append(_captioned(_image_at(failure_policy, view, c), f"F policy\nstep {c}", width=image_width))

    success_policy_ref = dict(success_target)
    success_policy_ref["obs_step"] = c
    success_policy_ref["state_step"] = c
    cells.append(_captioned(_image_at(success_policy_ref, view, c), f"S policy\nstep {c}", width=image_width))

    for dt in range(-context, context + 1):
        step = t + dt
        cells.append(_captioned(_image_at(failure_target, view, step), f"F target{dt:+d}\nstep {step}", width=image_width))
    for dt in range(-context, context + 1):
        step = t + dt
        cells.append(_captioned(_image_at(success_target, view, step), f"S target{dt:+d}\nstep {step}", width=image_width))

    sheet = _concat_grid(cells, cols=2 + (2 * context + 1), gap=8)
    header_h = 84
    out = Image.new("RGB", (sheet.width, sheet.height + header_h), "white")
    out.paste(sheet, (0, header_h))
    draw = ImageDraw.Draw(out)
    debug = row.get("selection_debug", {})
    idm = _idm_stats(row, idm_actions)
    title = (
        f"row={row_idx} {row['suite']} task={row['task_id']} ep={row['episode_idx']} "
        f"S={row['success_seed']} F={row['failure_seed']} view={view}"
    )
    line2 = (
        f"policy={c} target={t} offset={offset} chunk={row.get('chunk_size')} "
        f"score={float(debug.get('target_score', 0.0)):.3f} "
        f"state_growth={float(debug.get('target_state_growth', 0.0)):.3f} "
        f"action_gap={float(debug.get('action_gap_at_target', 0.0)):.3f}"
    )
    line3 = (
        f"idm_l2={idm['idm_l2']:.3f} pos={idm['idm_pos_l2']:.3f} "
        f"rot={idm['idm_rot_l2']:.3f} grip={idm['idm_grip_l2']:.3f}"
        if idm
        else "idm: not provided"
    )
    draw.text((8, 8), title, fill="black", font=_font(14))
    draw.text((8, 32), line2, fill="black", font=_font(13))
    draw.text((8, 56), line3, fill="black", font=_font(13))

    name = f"row_{row_idx:06d}_{row['suite']}_task{row['task_id']}_ep{row['episode_idx']}_{view}.jpg"
    path = output_dir / name
    out.save(path, quality=92)
    result = {
        "row": row_idx,
        "path": str(path),
        "suite": row["suite"],
        "task_id": row["task_id"],
        "episode_idx": row["episode_idx"],
        "success_seed": row["success_seed"],
        "failure_seed": row["failure_seed"],
        "view": view,
        "policy_obs_step": c,
        "target_step": t,
        "action_offset": offset,
        "target_score": float(debug.get("target_score", 0.0)),
        "target_state_growth": float(debug.get("target_state_growth", 0.0)),
        "action_gap_at_target": float(debug.get("action_gap_at_target", 0.0)),
    }
    if idm:
        result.update(idm)
    return result


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = _load_rows(Path(args.pairs))
    selected = _select_rows(rows, mode=args.selection, n=args.num_samples, seed=args.seed)
    views = args.view or ["agentview"]
    idm_actions = None
    if args.idm_actions:
        with np.load(args.idm_actions) as data:
            idm_actions = data["idm_action"].astype(np.float32)

    written = []
    for row_idx, row in selected:
        for view in views:
            try:
                written.append(
                    _make_sheet(
                        row_idx=row_idx,
                        row=row,
                        view=view,
                        output_dir=output_dir,
                        context=args.context,
                        image_width=args.image_width,
                        idm_actions=idm_actions,
                    )
                )
            except Exception as exc:
                print(f"[warn] row={row_idx} view={view}: {exc}", file=sys.stderr)

    summary = {
        "pairs": str(Path(args.pairs).resolve()),
        "idm_actions": str(Path(args.idm_actions).resolve()) if args.idm_actions else None,
        "selection": args.selection,
        "num_requested": args.num_samples,
        "num_written": len(written),
        "views": views,
        "rows": written,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Wrote {len(written)} inspection sheets to {output_dir}")


if __name__ == "__main__":
    main()
