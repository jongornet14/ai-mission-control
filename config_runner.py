#!/usr/bin/env python3
"""
Simple Config Runner - Launch experiments using config files
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


def colored(text: str, color: str) -> str:
    """Simple colored output"""
    colors = {
        "green": "\033[92m",
        "cyan": "\033[96m", 
        "yellow": "\033[93m",
        "blue": "\033[94m",
        "red": "\033[91m",
        "reset": "\033[0m",
    }
    return f"{colors.get(color, colors['reset'])}{text}{colors['reset']}"


def find_config_files(config_dir: str = "configs") -> List[Path]:
    """Find all config files in directory"""
    config_path = Path(config_dir)
    if not config_path.exists():
        return []
    
    return list(config_path.glob("*.json"))


def preview_config(config_path: Path) -> Dict:
    """Preview config file contents"""
    try:
        with open(config_path) as f:
            config = json.load(f)
        return config
    except Exception as e:
        print(colored(f"Error reading {config_path}: {e}", "red"))
        return {}


def list_configs(config_dir: str = "configs"):
    """List available config files with summaries"""
    configs = find_config_files(config_dir)
    
    if not configs:
        print(colored(f"No config files found in {config_dir}/", "yellow"))
        return
    
    print(colored(f"=== Available Config Files in {config_dir}/ ===", "green"))
    print()
    
    for config_file in sorted(configs):
        config = preview_config(config_file)
        if not config:
            continue
            
        name = config.get("experiment", {}).get("name", "Unnamed")
        desc = config.get("experiment", {}).get("description", "No description")
        algo = config.get("algorithm", {}).get("name", "unknown")
        env = config.get("environment", {}).get("name", "unknown")
        episodes = config.get("training", {}).get("max_episodes", "unknown")
        
        print(colored(f"📄 {config_file.name}", "cyan"))
        print(colored(f"   Experiment: {name}", "blue"))
        print(colored(f"   Description: {desc}", "blue"))
        print(colored(f"   Algorithm: {algo} | Environment: {env} | Episodes: {episodes}", "yellow"))
        print()


def validate_config_file(config_path: Path):
    """Validate a config file using the worker script"""
    cmd = [
        sys.executable, "scripts/worker_entry.py",
        "--config", str(config_path),
        "--validate-config"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(colored(f"✓ {config_path.name} is valid", "green"))
        return True
    else:
        print(colored(f"✗ {config_path.name} has errors:", "red"))
        print(result.stderr)
        return False


def run_experiment(config_path: Path, worker_id: int, **overrides):
    """Run an experiment with the given config"""
    cmd = [
        sys.executable, "scripts/worker_entry.py",
        "--config", str(config_path),
        "--worker_id", str(worker_id)
    ]
    
    # Add overrides
    for key, value in overrides.items():
        if value is not None:
            cmd.extend([f"--{key}", str(value)])
    
    print(colored(f"Running: {' '.join(cmd)}", "cyan"))
    print()
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(colored(f"Experiment failed with exit code {e.returncode}", "red"))
    except KeyboardInterrupt:
        print(colored("Experiment interrupted by user", "yellow"))


def main():
    parser = argparse.ArgumentParser(
        description="Config-driven experiment runner",
        epilog="""
Examples:
  # List available configs
  python config_runner.py --list

  # Validate a config file
  python config_runner.py --validate configs/sample_ppo_config.json

  # Run experiment with config file
  python config_runner.py --config configs/sample_ppo_config.json --worker_id 1

  # Run with device override
  python config_runner.py --config configs/sample_ddpg_config.json --worker_id 1 --device cpu
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("--config", type=str, help="Config file to run")
    parser.add_argument("--worker_id", type=int, help="Worker ID")
    parser.add_argument("--list", action="store_true", help="List available config files")
    parser.add_argument("--validate", type=str, help="Validate specific config file")
    parser.add_argument("--validate-all", action="store_true", help="Validate all config files")
    
    # Override options
    parser.add_argument("--device", type=str, help="Override device")
    parser.add_argument("--max_episodes", type=int, help="Override max episodes")
    parser.add_argument("--shared_dir", type=str, help="Override shared directory")
    
    args = parser.parse_args()
    
    if args.list:
        list_configs()
        return
    
    if args.validate:
        config_path = Path(args.validate)
        if not config_path.exists():
            print(colored(f"Config file not found: {config_path}", "red"))
            sys.exit(1)
        validate_config_file(config_path)
        return
    
    if args.validate_all:
        configs = find_config_files()
        print(colored("Validating all config files...", "green"))
        valid_count = 0
        for config_path in configs:
            if validate_config_file(config_path):
                valid_count += 1
        print(colored(f"\n{valid_count}/{len(configs)} config files are valid", "green"))
        return
    
    if args.config:
        config_path = Path(args.config)
        if not config_path.exists():
            print(colored(f"Config file not found: {config_path}", "red"))
            sys.exit(1)
        
        if not args.worker_id:
            print(colored("Error: --worker_id is required when running config", "red"))
            sys.exit(1)
        
        # Collect overrides
        overrides = {
            "device": args.device,
            "max_episodes": args.max_episodes,
            "shared_dir": args.shared_dir,
        }
        
        # Preview config before running
        config = preview_config(config_path)
        if config:
            name = config.get("experiment", {}).get("name", "Unnamed")
            print(colored(f"Running experiment: {name}", "green"))
            print(colored(f"Config file: {config_path}", "cyan"))
            print(colored(f"Worker ID: {args.worker_id}", "cyan"))
            print()
        
        run_experiment(config_path, args.worker_id, **overrides)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
