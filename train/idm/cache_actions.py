"""Cache goal-image IDM recovery action chunks for preference pairs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from train.idm import (  # noqa: E402
    NpzLRU,
    load_idm_config,
    load_goal_image_idm_checkpoint,
    predict_goal_image_idm_actions,
    resolve_idm_data_config,
)

DEFAULT_CONFIG = Path("experiment_specs/idm/libero_groot_goal_image_cache.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="YAML/JSON IDM cache config.")
    return parser.parse_args()


def _section(config: dict[str, object], key: str) -> dict[str, object]:
    value = config.get(key, {})
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise TypeError(f"Config section {key!r} must be a mapping")
    return value


def _list_or_none(value: object) -> list[str] | None:
    if value is None:
        return None
    if not isinstance(value, list):
        raise TypeError("Expected a list")
    return [str(v) for v in value]


def _device_from_config(value: object) -> str:
    if value is None or str(value) == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return str(value)


def _refs_from_row(row: dict[str, object]) -> tuple[dict[str, object], dict[str, object]]:
    if row.get("sample_type") == "chunk_preference_pair":
        return row["negative"], row["positive"]  # type: ignore[index,return-value]
    if "anchor" in row and "positive" in row:
        return row["anchor"], row["positive"]  # type: ignore[index,return-value]
    raise KeyError("Expected chunk_preference_pair negative/positive refs or legacy anchor/positive refs")


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    cfg = load_idm_config(config_path)
    paths = _section(cfg, "paths")
    data = _section(cfg, "data")
    cache_cfg = _section(cfg, "cache")
    runtime = _section(cfg, "runtime")

    output_dir = Path(str(paths["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(_device_from_config(runtime.get("device", "auto")))
    checkpoint_path = Path(str(paths["checkpoint"]))

    model, stats, config = load_goal_image_idm_checkpoint(checkpoint_path, map_location=device)
    model.to(device)
    model.eval()
    preset = str(data.get("preset", config.get("preset", "libero:groot")))
    image_key_override = _list_or_none(data.get("image_keys"))
    data_config = resolve_idm_data_config(
        preset,
        image_keys=tuple(image_key_override) if image_key_override else None,
        state_key=data.get("state_key"),
    )
    image_keys = tuple(image_key_override or config.get("image_keys") or data_config["image_keys"])
    state_key = str(data.get("state_key") or config.get("state_key") or data_config["state_key"])
    image_size = int(config["image_size"])
    horizon = int(config["horizon"])
    action_dim = int(config["action_dim"])
    clamp = bool(cache_cfg.get("clamp", True))

    rows: list[dict[str, object]] = []
    actions: list[np.ndarray] = []
    cache = NpzLRU(int(runtime.get("cache_size", 32)))
    pair_path = Path(str(paths["pairs"]))
    try:
        with pair_path.open() as f:
            for line_idx, line in enumerate(f):
                if not line.strip():
                    continue
                row = json.loads(line)
                current_ref, goal_ref = _refs_from_row(row)
                idm_action = predict_goal_image_idm_actions(
                    model,
                    stats,
                    current_ref,
                    goal_ref,
                    image_keys=image_keys,
                    image_size=image_size,
                    device=device,
                    npz_cache=cache,
                    state_key=state_key,
                )
                if clamp:
                    idm_action = np.clip(idm_action, -1.0, 1.0)
                actions.append(idm_action.astype(np.float32))
                row["idm"] = {
                    "type": "goal_image_idm",
                    "checkpoint": str(checkpoint_path.resolve()),
                    "cache": "idm_actions.npz",
                    "key": "idm_action",
                    "index": len(actions) - 1,
                    "horizon": horizon,
                    "source": "current_image_to_goal_image",
                    "current": "negative_target_observation",
                    "goal": "positive_target_observation",
                    "image_keys": list(image_keys),
                    "state_key": state_key,
                    "image_size": image_size,
                    "clamped": clamp,
                }
                if row.get("sample_type") == "chunk_preference_pair":
                    row["idm"]["source"] = "negative_target_observation_to_positive_target_observation"
                row["recover_positive"] = {
                    "type": "goal_image_idm_prefix",
                    "idm_index": len(actions) - 1,
                }
                row["cache_source_line"] = line_idx
                rows.append(row)
    finally:
        cache.close()

    action_arr = (
        np.stack(actions, axis=0)
        if actions
        else np.zeros((0, horizon, action_dim), dtype=np.float32)
    )
    np.savez_compressed(
        output_dir / "idm_actions.npz",
        idm_action=action_arr,
        checkpoint=np.asarray(str(checkpoint_path.resolve())),
        horizon=np.asarray(horizon, dtype=np.int32),
        image_keys=np.asarray(image_keys),
        clamped=np.asarray(clamp),
    )

    out_manifest = output_dir / "preference_pairs_with_idm.jsonl"
    with out_manifest.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    flat = action_arr.reshape(-1, action_dim) if len(action_arr) else action_arr
    summary = {
        "input_pairs": str(pair_path.resolve()),
        "checkpoint": str(checkpoint_path.resolve()),
        "config_path": str(config_path),
        "output_manifest": str(out_manifest),
        "cache": str(output_dir / "idm_actions.npz"),
        "rows": len(rows),
        "action_shape": list(action_arr.shape),
        "action_min": flat.min(axis=0).astype(float).tolist() if len(flat) else None,
        "action_max": flat.max(axis=0).astype(float).tolist() if len(flat) else None,
        "idm_config": config,
        "preset": preset,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Wrote {len(rows)} goal-image IDM cached rows: {out_manifest}")
    print(f"IDM actions: {output_dir / 'idm_actions.npz'} shape={action_arr.shape}")


if __name__ == "__main__":
    main()
