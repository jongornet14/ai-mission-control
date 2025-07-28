#!/usr/bin/env python3
"""
Worker Entry Script for Distributed Training
Uses simple inheritance: DistributedWorker(BaseWorker)
"""

import sys
import argparse
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, '/workspace/project')

# Import the distributed worker class
from intellinaut.workers.distributed import DistributedWorker

def main():
    """Main entry point for distributed worker"""
    parser = argparse.ArgumentParser(description='Distributed RL Worker (Simple Inheritance)')
    
    parser.add_argument('--worker_id', type=int, required=True, help='Worker ID')
    parser.add_argument('--shared_dir', type=str, required=True, help='Shared directory')
    parser.add_argument('--env', type=str, default='CartPole-v1', help='Environment name')
    parser.add_argument('--max_episodes', type=int, default=1000, help='Max episodes')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--sync_frequency', type=int, default=10, help='Episodes between sync checks')
    
    args = parser.parse_args()
    
    print(f"🎯 Starting Distributed Worker {args.worker_id}")
    print(f"📁 Shared directory: {args.shared_dir}")
    print(f"🎮 Environment: {args.env}")
    print(f"📊 Max episodes: {args.max_episodes}")
    print(f"🔄 Sync frequency: every {args.sync_frequency} episodes")
    
    # Create distributed worker using simple inheritance
    worker = DistributedWorker(
        worker_id=args.worker_id,
        shared_dir=args.shared_dir,
        max_episodes=args.max_episodes,
        checkpoint_frequency=50,
        status_update_frequency=10
    )
    
    # Run distributed training
    worker.run_distributed(
        env_name=args.env,
        device=args.device,
        lr=args.lr,
        sync_check_frequency=args.sync_frequency
    )


if __name__ == "__main__":
    main()