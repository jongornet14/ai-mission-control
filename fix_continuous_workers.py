#!/usr/bin/env python3
"""
Fix for making distributed workers run continuously instead of restarting
"""

import re
from pathlib import Path

def make_workers_run_continuously():
    """
    Modify distributed_worker.py to run continuously instead of finishing
    """
    worker_file = Path("scripts/distributed_worker.py")
    
    if not worker_file.exists():
        print(f"❌ File not found: {worker_file}")
        return False
    
    with open(worker_file, 'r') as f:
        content = f.read()
    
    # Find and replace the main function section
    old_main_section = '''def main():
    """Simplified main entry point - clean single responsibility"""
    parser = argparse.ArgumentParser(description='Enhanced Distributed RL Worker')
    
    # Worker-specific args
    parser.add_argument('--worker_id', type=int, required=True, help='Worker ID (0-19)')
    parser.add_argument('--shared_dir', type=str, required=True, help='Shared directory')
    parser.add_argument('--max_episodes', type=int, default=100, help='Episodes per checkpoint')
    parser.add_argument('--sync_interval', type=int, default=10, help='Episodes between syncs')
    parser.add_argument('--timeout_minutes', type=int, default=30, help='Max minutes before sync')
    
    # RL training args
    parser.add_argument('--env', type=str, default='CartPole-v1', help='Environment name')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    
    args = parser.parse_args()
    
    # Validate worker_id
    if args.worker_id < 0 or args.worker_id >= 20:
        raise ValueError("worker_id must be between 0 and 19")
    
    # Create worker and run training
    worker = DistributedWorker(
        worker_id=args.worker_id,
        shared_dir=args.shared_dir,
        max_episodes=args.max_episodes,
        sync_interval=args.sync_interval,
        timeout_minutes=args.timeout_minutes
    )
    
    # Single, clean call to run training
    worker.run(args)


if __name__ == "__main__":
    main()'''

    new_main_section = '''def main():
    """Simplified main entry point - clean single responsibility"""
    parser = argparse.ArgumentParser(description='Enhanced Distributed RL Worker')
    
    # Worker-specific args
    parser.add_argument('--worker_id', type=int, required=True, help='Worker ID (0-19)')
    parser.add_argument('--shared_dir', type=str, required=True, help='Shared directory')
    parser.add_argument('--max_episodes', type=int, default=100, help='Episodes per checkpoint')
    parser.add_argument('--sync_interval', type=int, default=10, help='Episodes between syncs')
    parser.add_argument('--timeout_minutes', type=int, default=30, help='Max minutes before sync')
    
    # RL training args
    parser.add_argument('--env', type=str, default='CartPole-v1', help='Environment name')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    
    args = parser.parse_args()
    
    # Validate worker_id
    if args.worker_id < 0 or args.worker_id >= 20:
        raise ValueError("worker_id must be between 0 and 19")
    
    # Run continuous training cycles
    cycle = 0
    while True:
        cycle += 1
        print(f"🔄 Worker {args.worker_id}: Starting training cycle {cycle}")
        
        try:
            # Create worker and run training
            worker = DistributedWorker(
                worker_id=args.worker_id,
                shared_dir=args.shared_dir,
                max_episodes=args.max_episodes,
                sync_interval=args.sync_interval,
                timeout_minutes=args.timeout_minutes
            )
            
            # Run one training cycle
            worker.run(args)
            
            # Brief pause between cycles
            print(f"✅ Worker {args.worker_id}: Completed cycle {cycle}, starting next cycle in 10 seconds...")
            time.sleep(10)
            
        except KeyboardInterrupt:
            print(f"🛑 Worker {args.worker_id}: Stopping continuous training")
            break
        except Exception as e:
            print(f"❌ Worker {args.worker_id}: Error in cycle {cycle}: {e}")
            print("💤 Waiting 30 seconds before retrying...")
            time.sleep(30)  # Wait before restarting


if __name__ == "__main__":
    main()'''
    
    # Replace the main section
    if old_main_section in content:
        content = content.replace(old_main_section, new_main_section)
        print("✅ Found and replaced main function with continuous version")
    else:
        # Fallback: just replace the if __name__ == "__main__": section
        old_if_main = '''if __name__ == "__main__":
    main()'''
        
        new_if_main = '''def main_continuous():
    """Main entry point with continuous training cycles"""
    parser = argparse.ArgumentParser(description='Enhanced Distributed RL Worker')
    
    # Worker-specific args
    parser.add_argument('--worker_id', type=int, required=True, help='Worker ID (0-19)')
    parser.add_argument('--shared_dir', type=str, required=True, help='Shared directory')
    parser.add_argument('--max_episodes', type=int, default=100, help='Episodes per checkpoint')
    parser.add_argument('--sync_interval', type=int, default=10, help='Episodes between syncs')
    parser.add_argument('--timeout_minutes', type=int, default=30, help='Max minutes before sync')
    
    # RL training args
    parser.add_argument('--env', type=str, default='CartPole-v1', help='Environment name')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    
    args = parser.parse_args()
    
    # Validate worker_id
    if args.worker_id < 0 or args.worker_id >= 20:
        raise ValueError("worker_id must be between 0 and 19")
    
    # Run continuous training cycles
    cycle = 0
    while True:
        cycle += 1
        print(f"🔄 Worker {args.worker_id}: Starting training cycle {cycle}")
        
        try:
            # Create worker and run training
            worker = DistributedWorker(
                worker_id=args.worker_id,
                shared_dir=args.shared_dir,
                max_episodes=args.max_episodes,
                sync_interval=args.sync_interval,
                timeout_minutes=args.timeout_minutes
            )
            
            # Run one training cycle
            worker.run(args)
            
            # Brief pause between cycles
            print(f"✅ Worker {args.worker_id}: Completed cycle {cycle}, starting next cycle in 10 seconds...")
            time.sleep(10)
            
        except KeyboardInterrupt:
            print(f"🛑 Worker {args.worker_id}: Stopping continuous training")
            break
        except Exception as e:
            print(f"❌ Worker {args.worker_id}: Error in cycle {cycle}: {e}")
            print("💤 Waiting 30 seconds before retrying...")
            time.sleep(30)  # Wait before restarting


if __name__ == "__main__":
    main_continuous()'''
        
        content = content.replace(old_if_main, new_if_main)
        print("✅ Replaced if __name__ section with continuous version")
    
    # Write back the modified content
    with open(worker_file, 'w') as f:
        f.write(content)
    
    print(f"✅ Modified {worker_file} for continuous training")
    return True


def test_continuous_worker():
    """Test the continuous worker with a single instance"""
    import subprocess
    
    print("🧪 Testing continuous worker setup...")
    
    # Test command
    cmd = [
        "docker", "run", "-it", "--rm", "--gpus", "all",
        "-v", f"{Path.cwd()}:/workspace/project",
        "-v", f"{Path.cwd()}/distributed_shared:/workspace/distributed_shared",
        "ai-mission-control:latest",
        "timeout", "60",  # Run for 60 seconds to test
        "python", "scripts/distributed_worker.py",
        "--worker_id", "99",  # Test worker ID
        "--shared_dir", "/workspace/distributed_shared",
        "--env", "CartPole-v1",
        "--max_episodes", "10"  # Small number for testing
    ]
    
    print("Running test command:")
    print(" ".join(cmd))
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=70)
        print("\n📋 Test Output:")
        print(result.stdout)
        if result.stderr:
            print("\n⚠️ Test Errors:")
            print(result.stderr)
    except subprocess.TimeoutExpired:
        print("✅ Test completed (timeout as expected)")
    except Exception as e:
        print(f"❌ Test failed: {e}")


if __name__ == "__main__":
    print("🔄 Making workers run continuously...")
    
    if make_workers_run_continuously():
        print("\n✅ Workers modified for continuous training!")
        print("\n🚀 Next steps:")
        print("1. Stop current distributed training:")
        print("   make distributed-down")
        print("\n2. Start new continuous training:")
        print("   make distributed-up")
        print("\n3. Monitor progress:")
        print("   make distributed-logs")
        print("   make distributed-tensorboard")
        
        print("\n🔬 Optional: Test the changes first:")
        print("   python fix_continuous_workers.py  # Run test_continuous_worker()")
    else:
        print("❌ Failed to modify workers")