#!/usr/bin/env python3
"""
Distributed Training CLI - Simple Inheritance Version
Manage distributed RL training with 2 workers + coordinator
Uses DistributedWorker(BaseWorker) - clean and straightforward
"""

import os
import sys
import json
import time
import subprocess
import argparse
from pathlib import Path


class DistributedCLI:
    """CLI for managing distributed training using simple inheritance"""
    
    def __init__(self):
        self.project_root = Path.cwd()
        self.shared_dir = self.project_root / "distributed_shared"
        self.compose_file = "docker-compose.distributed.yml"
        print(f"🎯 Distributed CLI initialized")
        print(f"📁 Project root: {self.project_root}")
        print(f"🔗 Using simple inheritance: DistributedWorker(BaseWorker)")
    
    def generate_compose_file(self, num_workers=2, env="CartPole-v1"):
        """Generate docker-compose file for distributed training"""
        print(f"📝 Generating {self.compose_file} with {num_workers} workers...")
        
        # Import the generator function
        sys.path.insert(0, str(self.project_root))
        try:
            from scripts.generate_docker_compose_distributed import generate_docker_compose_distributed
            generate_docker_compose_distributed(num_workers, self.compose_file)
            print(f"✅ Generated {self.compose_file}")
            return True
        except ImportError as e:
            print(f"❌ Could not import generator: {e}")
            print(f"💡 Make sure scripts/generate_docker_compose_distributed.py exists")
            return False
    
    def start_training(self, workers=2, env="CartPole-v1", force=False):
        """Start distributed training"""
        print(f"🚀 Starting distributed training...")
        print(f"   Workers: {workers}")
        print(f"   Environment: {env}")
        print(f"   Architecture: DistributedWorker(BaseWorker)")
        
        # Check if already running
        if not force and self._is_running():
            print("⚠️  Training already running. Use --force to restart.")
            return False
        
        # Generate compose file
        if not self.generate_compose_file(workers, env):
            return False
        
        # Create shared directory
        self.shared_dir.mkdir(exist_ok=True)
        print(f"📂 Created shared directory: {self.shared_dir}")
        
        # Start containers
        try:
            cmd = [
                "docker-compose", "-f", self.compose_file, "up", "-d",
                "--remove-orphans"
            ]
            
            env_vars = os.environ.copy()
            env_vars["ENV"] = env
            
            result = subprocess.run(cmd, env=env_vars, capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ Distributed training started successfully!")
                print("⏳ Waiting for containers to initialize...")
                time.sleep(5)  # Let containers initialize
                self.status()
                return True
            else:
                print(f"❌ Failed to start training:")
                print(result.stderr)
                return False
                
        except FileNotFoundError:
            print("❌ docker-compose not found. Please install Docker Compose.")
            return False
        except Exception as e:
            print(f"❌ Error starting training: {e}")
            return False
    
    def stop_training(self):
        """Stop distributed training"""
        print("🛑 Stopping distributed training...")
        
        try:
            cmd = ["docker-compose", "-f", self.compose_file, "down"]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ Distributed training stopped")
                return True
            else:
                print(f"❌ Error stopping training:")
                print(result.stderr)
                return False
                
        except Exception as e:
            print(f"❌ Error stopping training: {e}")
            return False
    
    def status(self):
        """Show comprehensive training status"""
        print("📊 Distributed Training Status")
        print("=" * 50)
        
        # Check containers
        try:
            cmd = ["docker-compose", "-f", self.compose_file, "ps"]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print("🐳 Container Status:")
                lines = result.stdout.strip().split('\n')
                if len(lines) > 1:  # Has actual containers
                    for line in lines:
                        print(f"   {line}")
                else:
                    print("   No containers running")
            else:
                print("❌ No containers running or compose file not found")
        except Exception as e:
            print(f"❌ Error checking containers: {e}")
        
        # Check coordinator status
        self._show_coordinator_status()
        
        # Check worker status
        self._show_worker_status()
        
        # Show shared directory info
        self._show_shared_directory_info()
    
    def _show_coordinator_status(self):
        """Show coordinator status details"""
        status_file = self.shared_dir / "coordinator_status.json"
        print(f"\n🎯 Coordinator Status:")
        
        if status_file.exists():
            try:
                with open(status_file) as f:
                    coordinator_status = json.load(f)
                
                print(f"   Status: {coordinator_status.get('status', 'UNKNOWN')}")
                print(f"   Sync Count: {coordinator_status.get('sync_count', 0)}")
                print(f"   Uptime: {coordinator_status.get('uptime_seconds', 0)}s")
                print(f"   Managing: {coordinator_status.get('num_workers', 0)} workers")
                
                last_heartbeat = coordinator_status.get('last_heartbeat')
                if last_heartbeat:
                    print(f"   Last Heartbeat: {last_heartbeat}")
                    
            except Exception as e:
                print(f"   Error reading coordinator status: {e}")
        else:
            print("   No coordinator status found (may still be starting)")
    
    def _show_worker_status(self):
        """Show worker status details"""
        worker_logs_dir = self.shared_dir / "worker_logs"
        print(f"\nWorker Status:")
        
        if worker_logs_dir.exists():
            workers = list(worker_logs_dir.glob("worker_*"))
            print(f"   Found {len(workers)} worker directories")
            
            for worker_dir in sorted(workers):
                worker_id = worker_dir.name.split('_')[1]
                status_file = worker_dir / f"worker_{worker_id}_status.json"
                
                if status_file.exists():
                    try:
                        with open(status_file) as f:
                            worker_status = json.load(f)
                        
                        status = worker_status.get('status', 'UNKNOWN')
                        episode = worker_status.get('episode', 0)
                        avg_reward = worker_status.get('avg_reward', 0)
                        
                        print(f"   Worker {worker_id}: {status} "
                              f"(Episode {episode}, Avg Reward: {avg_reward:.2f})")
                    except Exception as e:
                        print(f"   Worker {worker_id}: Error reading status - {e}")
                else:
                    print(f"   Worker {worker_id}: No status file (may be starting)")
        else:
            print("   No worker logs directory found")
    
    def _show_shared_directory_info(self):
        """Show shared directory structure and contents"""
        print(f"\nShared Directory Info:")
        
        if self.shared_dir.exists():
            print(f"   Location: {self.shared_dir}")
            
            # Count files in key directories
            subdirs = ['models', 'metrics', 'best_model', 'signals']
            for subdir in subdirs:
                dir_path = self.shared_dir / subdir
                if dir_path.exists():
                    file_count = len(list(dir_path.glob("*")))
                    print(f"   {subdir}/: {file_count} files")
                else:
                    print(f"   {subdir}/: Not created yet")
        else:
            print("   Shared directory doesn't exist")
    
    def logs(self, service=None, follow=False, tail=50):
        """Show logs from services"""
        if service:
            print(f"Showing logs for {service}...")
        else:
            print("Showing logs from all services...")
        
        try:
            cmd = ["docker-compose", "-f", self.compose_file, "logs"]
            
            if not follow:
                cmd.extend(["--tail", str(tail)])
            
            if follow:
                cmd.append("-f")
            
            if service:
                cmd.append(service)
            
            # Run interactively for log following
            if follow:
                print("💡 Press Ctrl+C to stop following logs")
                subprocess.run(cmd)
            else:
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.stdout:
                    print(result.stdout)
                if result.stderr:
                    print("STDERR:", result.stderr)
                    
        except KeyboardInterrupt:
            print("\nLog following stopped")
        except Exception as e:
            print(f"Error showing logs: {e}")
    
    def cleanup(self):
        """Clean up all resources"""
        print("Cleaning up distributed training...")
        
        # Stop containers
        self.stop_training()
        
        # Remove shared directory
        import shutil
        if self.shared_dir.exists():
            try:
                shutil.rmtree(self.shared_dir)
                print(f"Removed {self.shared_dir}")
            except Exception as e:
                print(f"Error removing shared directory: {e}")
        
        # Remove compose file
        if Path(self.compose_file).exists():
            try:
                Path(self.compose_file).unlink()
                print(f"Removed {self.compose_file}")
            except Exception as e:
                print(f"Error removing compose file: {e}")
        
        print("Cleanup completed")
    
    def _is_running(self):
        """Check if training is currently running"""
        try:
            cmd = ["docker-compose", "-f", self.compose_file, "ps", "-q"]
            result = subprocess.run(cmd, capture_output=True, text=True)
            return bool(result.stdout.strip())
        except:
            return False


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description='Distributed RL Training CLI (Simple Inheritance)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s start                              # Start with 2 workers, CartPole-v1
  %(prog)s start --workers 4 --env Ant-v4    # Start with 4 workers, Ant environment
  %(prog)s status                             # Check training status
  %(prog)s logs --follow                      # Follow all logs
  %(prog)s logs --service coordinator         # Show coordinator logs only
  %(prog)s stop                               # Stop training
  %(prog)s cleanup                            # Remove all files and containers
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Start command
    start_parser = subparsers.add_parser('start', help='Start distributed training')
    start_parser.add_argument('--workers', type=int, default=2, help='Number of workers (default: 2)')
    start_parser.add_argument('--env', type=str, default='CartPole-v1', help='Environment (default: CartPole-v1)')
    start_parser.add_argument('--force', action='store_true', help='Force restart if already running')
    
    # Stop command
    subparsers.add_parser('stop', help='Stop distributed training')
    
    # Status command
    subparsers.add_parser('status', help='Show comprehensive training status')
    
    # Logs command
    logs_parser = subparsers.add_parser('logs', help='Show logs from services')
    logs_parser.add_argument('--service', type=str, 
                           help='Specific service (coordinator, worker-0, worker-1, etc.)')
    logs_parser.add_argument('--follow', '-f', action='store_true', help='Follow logs in real-time')
    logs_parser.add_argument('--tail', type=int, default=50, help='Number of lines to show (default: 50)')
    
    # Cleanup command
    subparsers.add_parser('cleanup', help='Clean up all resources (containers, files, directories)')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    cli = DistributedCLI()
    
    try:
        if args.command == 'start':
            success = cli.start_training(args.workers, args.env, args.force)
            if success:
                print(f"\nNext steps:")
                print(f"   python scripts/distributed_cli.py status    # Check progress")
                print(f"   python scripts/distributed_cli.py logs -f   # Follow logs")
        elif args.command == 'stop':
            cli.stop_training()
        elif args.command == 'status':
            cli.status()
        elif args.command == 'logs':
            cli.logs(args.service, args.follow, args.tail)
        elif args.command == 'cleanup':
            cli.cleanup()
    except KeyboardInterrupt:
        print(f"\nCommand interrupted")
    except Exception as e:
        print(f"Unexpected error: {e}")


if __name__ == "__main__":
    main()