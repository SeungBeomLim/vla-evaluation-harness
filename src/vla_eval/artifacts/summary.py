"""Human-readable and flat result summaries."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


def _episodes(result: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    out = []
    for task in result.get("tasks", []):
        task_name = task.get("task", "")
        for ep in task.get("episodes", []):
            out.append((task_name, ep))
    return out


def _success(ep: dict[str, Any]) -> bool:
    return bool(ep.get("metrics", {}).get("success", False))


def _pct(num: int | float, den: int | float) -> float:
    return float(num) / float(den) * 100.0 if den else 0.0


def _fmt_duration(seconds: float | None) -> str:
    if seconds is None:
        return "unknown"
    seconds = max(0.0, float(seconds))
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:05.2f}"


def _model_label(result: dict[str, Any]) -> str:
    info = result.get("server_info") or {}
    server = info.get("model_server", "unknown")
    return str(server)


def write_episodes_jsonl(result: dict[str, Any], output_dir: Path, safe_name: str) -> Path:
    """Write flat episode records for easy downstream loading."""
    path = output_dir / f"{safe_name}_episodes.jsonl"
    with path.open("w") as f:
        for task_name, ep in _episodes(result):
            metrics = ep.get("metrics", {})
            record = {
                "benchmark": result.get("benchmark"),
                "task": task_name,
                "episode_id": ep.get("episode_id"),
                "success": bool(metrics.get("success", False)),
                "steps": ep.get("steps"),
                "elapsed_sec": ep.get("elapsed_sec"),
                "failure_reason": ep.get("failure_reason"),
                "failure_detail": ep.get("failure_detail"),
                "metrics": metrics,
                "artifacts": ep.get("artifacts", {}),
            }
            f.write(json.dumps(record, default=str) + "\n")
    return path


def _calvin_lines(episodes: list[tuple[str, dict[str, Any]]]) -> list[str]:
    completed = [int(ep.get("metrics", {}).get("completed_subtasks", 0)) for _, ep in episodes]
    total = len(completed)
    if not total:
        return []

    lines = ["Benchmark Metrics:"]
    avg_len = sum(completed) / total
    lines.append(f"  - Avg Completed Subtasks: {avg_len:.2f} / 5")
    for k in range(1, 6):
        succ = sum(1 for c in completed if c >= k)
        lines.append(f"  - Step {k}/5 Success: {succ}/{total} ({_pct(succ, total):.2f}%)")
    lines.append("-----------------------------------------")
    return lines


def write_experiment_summary(result: dict[str, Any], output_dir: Path, safe_name: str) -> Path:
    """Write a human-readable experiment summary text file."""
    path = output_dir / f"{safe_name}_summary.txt"
    episodes = _episodes(result)
    total = len(episodes)
    successes = sum(1 for _, ep in episodes if _success(ep))
    elapsed = sum(float(ep.get("elapsed_sec") or 0.0) for _, ep in episodes)

    created = result.get("created_at")
    try:
        created_text = datetime.fromisoformat(str(created)).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        created_text = str(created or "unknown")

    lines = [
        "=========================================",
        "             EXPERIMENT SUMMARY          ",
        "=========================================",
        f"Date: {created_text}",
        f"Benchmark: {result.get('benchmark', 'unknown')}",
        f"Model Server: {_model_label(result)}",
        f"Mode: {result.get('mode', 'unknown')}",
        f"Random Seed: {result.get('seed', 'unknown')}",
        f"Execution Time: {_fmt_duration(elapsed)}",
        "-----------------------------------------",
        f"Total Success Rate: {successes}/{total} ({_pct(successes, total):.2f}%)",
        "-----------------------------------------",
    ]

    if str(result.get("benchmark", "")).lower().startswith("calvin"):
        lines.extend(_calvin_lines(episodes))
    else:
        metric_keys = result.get("metric_keys", {})
        extra_metrics = []
        for key, agg in metric_keys.items():
            if key == "success":
                continue
            result_key = f"{agg}_{key}"
            if result_key in result:
                extra_metrics.append(f"  - {result_key}: {result[result_key]}")
        if extra_metrics:
            lines.append("Benchmark Metrics:")
            lines.extend(extra_metrics)
            lines.append("-----------------------------------------")

    lines.append("Per-Task Success Rates:")
    for task in result.get("tasks", []):
        eps = task.get("episodes", [])
        n = len(eps)
        succ = sum(1 for ep in eps if _success(ep))
        lines.append(f"  - {task.get('task', '')}: {succ}/{n} ({_pct(succ, n):.2f}%)")
    lines.append("=========================================")
    path.write_text("\n".join(lines) + "\n")
    return path
