#!/usr/bin/env python3
"""
Generate Docker Compose override for GPU-distributed workers
Usage: python scripts/generate_docker_compose_distributed.py --workers 12 --gpus 0,1,2,3
"""

import argparse
import yaml
import sys
from pathlib import Path


def generate_worker_overrides(num_workers: int, gpu_list: list) -> dict:
    """Generate docker-compose override for distributed workers"""

    # Base override structure
    override = {"version": "3.8", "services": {}}

    num_gpus = len(gpu_list)

    for worker_id in range(num_workers):
        # Round-robin GPU assignment
        gpu_index = worker_id % num_gpus
        assigned_gpu = str(gpu_list[gpu_index])

        # Generate worker service name
        service_name = f"worker-{worker_id}"

        # Worker configuration
        worker_config = {
            "extends": {"service": "worker", "file": "docker-compose.scalable.yml"},
            "container_name": f"rl-worker-{worker_id}",
            "environment": [
                f"CUDA_VISIBLE_DEVICES={assigned_gpu}",
                f"WORKER_ID={worker_id}",
                "PYTHONPATH=/workspace/project",
            ],
            "deploy": {
                "resources": {
                    "reservations": {
                        "devices": [
                            {
                                "driver": "nvidia",
                                "device_ids": [assigned_gpu],
                                "capabilities": ["gpu"],
                            }
                        ]
                    }
                }
            },
        }

        override["services"][service_name] = worker_config

    return override


def main():
    parser = argparse.ArgumentParser(
        description="Generate distributed workers docker-compose override"
    )
    parser.add_argument("--workers", type=int, required=True, help="Number of workers")
    parser.add_argument(
        "--gpus", type=str, required=True, help="Comma-separated GPU IDs"
    )
    parser.add_argument(
        "--output", type=str, default="docker-compose.override.yml", help="Output file"
    )

    args = parser.parse_args()

    # Parse GPU list
    gpu_list = [gpu.strip() for gpu in args.gpus.split(",")]

    print(f"Generating {args.workers} workers across GPUs: {gpu_list}")

    # Generate override configuration
    override_config = generate_worker_overrides(args.workers, gpu_list)

    # Write to file
    output_path = Path(args.output)
    with open(output_path, "w") as f:
        yaml.safe_dump(override_config, f, default_flow_style=False, indent=2)

    print(f"Generated override file: {output_path}")
    print("\nTo use this configuration:")
    print(f"docker-compose -f docker-compose.scalable.yml -f {output_path} up -d")

    return 0


if __name__ == "__main__":
    sys.exit(main())
