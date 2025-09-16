"""
Log Analysis and Monitoring Tools for Distributed RL Training
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import pandas as pd
import re


class LogAnalyzer:
    """Analyze structured logs from distributed training"""

    def __init__(self, log_dir: str = "distributed_shared"):
        self.log_dir = Path(log_dir)
        self.structured_dir = self.log_dir / "structured"

    def find_log_files(
        self, component: Optional[str] = None, worker_id: Optional[int] = None
    ) -> List[Path]:
        """Find relevant log files"""
        if not self.structured_dir.exists():
            return []

        pattern = "*.jsonl"
        if component:
            pattern = f"{component}*.jsonl"
        if worker_id is not None:
            pattern = f"*worker_{worker_id}*.jsonl"

        return list(self.structured_dir.glob(pattern))

    def load_logs(
        self, files: List[Path], since: Optional[datetime] = None
    ) -> List[Dict]:
        """Load and filter log entries"""
        entries = []

        for file_path in files:
            try:
                with open(file_path, "r") as f:
                    for line in f:
                        try:
                            entry = json.loads(line.strip())

                            # Filter by time if specified
                            if since:
                                entry_time = datetime.fromisoformat(
                                    entry["iso_timestamp"]
                                )
                                if entry_time < since:
                                    continue

                            entries.append(entry)
                        except json.JSONDecodeError:
                            continue
            except Exception as e:
                print(f"Error reading {file_path}: {e}")

        return sorted(entries, key=lambda x: x["timestamp"])

    def filter_by_event_type(
        self, entries: List[Dict], event_types: List[str]
    ) -> List[Dict]:
        """Filter entries by event type"""
        return [e for e in entries if e.get("event_type") in event_types]

    def filter_by_level(self, entries: List[Dict], levels: List[str]) -> List[Dict]:
        """Filter entries by log level"""
        return [e for e in entries if e.get("level") in levels]

    def get_error_summary(self, entries: List[Dict]) -> Dict:
        """Analyze errors and warnings"""
        errors = self.filter_by_level(entries, ["ERROR", "CRITICAL"])
        warnings = self.filter_by_level(entries, ["WARNING"])

        error_types = {}
        for entry in errors:
            error_type = entry.get("event_type", "unknown_error")
            if error_type not in error_types:
                error_types[error_type] = []
            error_types[error_type].append(entry)

        return {
            "total_errors": len(errors),
            "total_warnings": len(warnings),
            "error_types": error_types,
            "recent_errors": errors[-10:] if errors else [],
        }

    def get_training_progress(self, entries: List[Dict]) -> Dict:
        """Analyze training progress"""
        episode_entries = self.filter_by_event_type(entries, ["episode_end"])
        step_entries = self.filter_by_event_type(entries, ["step_completed"])

        # Group by worker
        worker_progress = {}
        for entry in episode_entries:
            worker_id = entry.get("worker_id", "unknown")
            if worker_id not in worker_progress:
                worker_progress[worker_id] = {
                    "episodes": [],
                    "rewards": [],
                    "latest_episode": 0,
                }

            if "episode" in entry and "total_reward" in entry:
                worker_progress[worker_id]["episodes"].append(entry["episode"])
                worker_progress[worker_id]["rewards"].append(entry["total_reward"])
                worker_progress[worker_id]["latest_episode"] = max(
                    worker_progress[worker_id]["latest_episode"], entry["episode"]
                )

        return worker_progress

    def get_performance_metrics(self, entries: List[Dict]) -> Dict:
        """Analyze system performance"""
        perf_entries = [
            e for e in entries if "system_info" in e or "performance_snapshot" in e
        ]

        cpu_usage = []
        memory_usage = []
        gpu_usage = []

        for entry in perf_entries:
            perf_data = entry.get("system_info") or entry.get(
                "performance_snapshot", {}
            )

            if "cpu_percent" in perf_data:
                cpu_usage.append(perf_data["cpu_percent"])
            if "memory_percent" in perf_data:
                memory_usage.append(perf_data["memory_percent"])

            for gpu_info in perf_data.get("gpu_info", []):
                if "gpu_util" in gpu_info:
                    gpu_usage.append(
                        {
                            "gpu_id": gpu_info["id"],
                            "utilization": gpu_info["gpu_util"],
                            "memory_percent": gpu_info["memory_percent"],
                        }
                    )

        return {
            "cpu_usage": {
                "avg": sum(cpu_usage) / len(cpu_usage) if cpu_usage else 0,
                "max": max(cpu_usage) if cpu_usage else 0,
                "samples": len(cpu_usage),
            },
            "memory_usage": {
                "avg": sum(memory_usage) / len(memory_usage) if memory_usage else 0,
                "max": max(memory_usage) if memory_usage else 0,
                "samples": len(memory_usage),
            },
            "gpu_usage": gpu_usage,
        }

    def get_hyperopt_progress(self, entries: List[Dict]) -> Dict:
        """Analyze hyperparameter optimization progress"""
        suggestions = self.filter_by_event_type(entries, ["hyperparam_suggestion"])
        results = self.filter_by_event_type(entries, ["hyperparam_result"])

        iterations = []
        scores = []
        best_score = float("-inf")
        best_params = None

        for entry in results:
            if "iteration" in entry and "score" in entry:
                iterations.append(entry["iteration"])
                score = entry["score"]
                scores.append(score)

                if score > best_score:
                    best_score = score
                    best_params = entry.get("hyperparams")

        return {
            "total_iterations": len(results),
            "best_score": best_score if best_score != float("-inf") else None,
            "best_params": best_params,
            "score_history": list(zip(iterations, scores)),
            "latest_suggestions": len(suggestions),
        }


def print_colored(text: str, color: str = "white") -> None:
    """Print colored text"""
    colors = {
        "red": "\033[91m",
        "green": "\033[92m",
        "yellow": "\033[93m",
        "blue": "\033[94m",
        "magenta": "\033[95m",
        "cyan": "\033[96m",
        "white": "\033[97m",
        "reset": "\033[0m",
    }
    print(f"{colors.get(color, colors['white'])}{text}{colors['reset']}")


def main():
    parser = argparse.ArgumentParser(description="Analyze distributed RL training logs")
    parser.add_argument("--log-dir", default="distributed_shared", help="Log directory")
    parser.add_argument("--component", help="Filter by component name")
    parser.add_argument("--worker", type=int, help="Filter by worker ID")
    parser.add_argument(
        "--since", help="Show logs since time (e.g., '2h', '30m', '1d')"
    )
    parser.add_argument(
        "--errors-only", action="store_true", help="Show only errors and warnings"
    )
    parser.add_argument(
        "--training-progress", action="store_true", help="Show training progress"
    )
    parser.add_argument(
        "--performance", action="store_true", help="Show performance metrics"
    )
    parser.add_argument(
        "--hyperopt",
        action="store_true",
        help="Show hyperparameter optimization progress",
    )
    parser.add_argument("--summary", action="store_true", help="Show overall summary")

    args = parser.parse_args()

    analyzer = LogAnalyzer(args.log_dir)

    # Parse time filter
    since = None
    if args.since:
        time_match = re.match(r"(\d+)([hmd])", args.since.lower())
        if time_match:
            value, unit = int(time_match.group(1)), time_match.group(2)
            if unit == "h":
                since = datetime.now() - timedelta(hours=value)
            elif unit == "m":
                since = datetime.now() - timedelta(minutes=value)
            elif unit == "d":
                since = datetime.now() - timedelta(days=value)

    # Load logs
    log_files = analyzer.find_log_files(args.component, args.worker)
    if not log_files:
        print_colored("No log files found", "red")
        return

    entries = analyzer.load_logs(log_files, since)
    if not entries:
        print_colored("No log entries found", "yellow")
        return

    print_colored(
        f"Loaded {len(entries)} log entries from {len(log_files)} files", "green"
    )

    # Show errors only
    if args.errors_only:
        error_summary = analyzer.get_error_summary(entries)

        print_colored("\n=== ERROR SUMMARY ===", "red")
        print(f"Total Errors: {error_summary['total_errors']}")
        print(f"Total Warnings: {error_summary['total_warnings']}")

        if error_summary["error_types"]:
            print_colored("\nError Types:", "yellow")
            for error_type, error_list in error_summary["error_types"].items():
                print(f"  {error_type}: {len(error_list)} occurrences")

        if error_summary["recent_errors"]:
            print_colored("\nRecent Errors:", "yellow")
            for error in error_summary["recent_errors"][-5:]:
                timestamp = error["iso_timestamp"][:19]
                message = error["message"][:100]
                print(f"  [{timestamp}] {message}")

    # Show training progress
    if args.training_progress:
        progress = analyzer.get_training_progress(entries)

        print_colored("\n=== TRAINING PROGRESS ===", "green")
        for worker_id, worker_data in progress.items():
            if worker_data["episodes"]:
                latest_episode = worker_data["latest_episode"]
                avg_reward = sum(worker_data["rewards"]) / len(worker_data["rewards"])
                recent_reward = (
                    worker_data["rewards"][-1] if worker_data["rewards"] else 0
                )

                print(f"Worker {worker_id}:")
                print(f"  Latest Episode: {latest_episode}")
                print(f"  Average Reward: {avg_reward:.2f}")
                print(f"  Recent Reward: {recent_reward:.2f}")
                print(f"  Total Episodes: {len(worker_data['episodes'])}")

    # Show performance metrics
    if args.performance:
        perf_metrics = analyzer.get_performance_metrics(entries)

        print_colored("\n=== PERFORMANCE METRICS ===", "blue")

        cpu = perf_metrics["cpu_usage"]
        if cpu["samples"] > 0:
            print(
                f"CPU Usage: Avg {cpu['avg']:.1f}%, Max {cpu['max']:.1f}% ({cpu['samples']} samples)"
            )

        memory = perf_metrics["memory_usage"]
        if memory["samples"] > 0:
            print(
                f"Memory Usage: Avg {memory['avg']:.1f}%, Max {memory['max']:.1f}% ({memory['samples']} samples)"
            )

        gpu_usage = perf_metrics["gpu_usage"]
        if gpu_usage:
            print("GPU Usage:")
            for gpu in gpu_usage[-5:]:  # Show recent GPU usage
                print(
                    f"  GPU {gpu['gpu_id']}: {gpu['utilization']:.1f}% util, {gpu['memory_percent']:.1f}% memory"
                )

    # Show hyperparameter optimization
    if args.hyperopt:
        hyperopt = analyzer.get_hyperopt_progress(entries)

        print_colored("\n=== HYPERPARAMETER OPTIMIZATION ===", "magenta")
        print(f"Total Iterations: {hyperopt['total_iterations']}")
        print(f"Latest Suggestions: {hyperopt['latest_suggestions']}")

        if hyperopt["best_score"] is not None:
            print(f"Best Score: {hyperopt['best_score']:.4f}")
            if hyperopt["best_params"]:
                print("Best Parameters:")
                for param, value in hyperopt["best_params"].items():
                    print(f"  {param}: {value}")

    # Show overall summary
    if args.summary or not any(
        [args.errors_only, args.training_progress, args.performance, args.hyperopt]
    ):
        error_summary = analyzer.get_error_summary(entries)
        progress = analyzer.get_training_progress(entries)
        perf_metrics = analyzer.get_performance_metrics(entries)
        hyperopt = analyzer.get_hyperopt_progress(entries)

        print_colored("\n=== OVERALL SUMMARY ===", "cyan")

        # Time range
        if entries:
            start_time = datetime.fromisoformat(entries[0]["iso_timestamp"])
            end_time = datetime.fromisoformat(entries[-1]["iso_timestamp"])
            duration = end_time - start_time
            print(
                f"Time Range: {start_time.strftime('%Y-%m-%d %H:%M:%S')} to {end_time.strftime('%H:%M:%S')}"
            )
            print(f"Duration: {duration}")

        # Components and workers
        components = set(e.get("component") for e in entries if e.get("component"))
        workers = set(
            e.get("worker_id") for e in entries if e.get("worker_id") is not None
        )
        print(f"Components: {', '.join(sorted(components))}")
        if workers:
            print(f"Workers: {', '.join(map(str, sorted(workers)))}")

        # Quick stats
        print(f"Total Log Entries: {len(entries)}")
        print(f"Errors: {error_summary['total_errors']}")
        print(f"Warnings: {error_summary['total_warnings']}")

        if progress:
            total_episodes = sum(len(w["episodes"]) for w in progress.values())
            print(f"Total Episodes Completed: {total_episodes}")

        if hyperopt["total_iterations"] > 0:
            print(f"Hyperopt Iterations: {hyperopt['total_iterations']}")
            if hyperopt["best_score"] is not None:
                print(f"Best Score: {hyperopt['best_score']:.4f}")


if __name__ == "__main__":
    main()
