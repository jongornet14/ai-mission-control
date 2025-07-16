#!/usr/bin/env python3
"""
Smart Distributed RL Launcher
Automatically detects GPU memory and spawns optimal number of workers
"""

import torch
import subprocess
import multiprocessing as mp
import time
import signal
import sys
import argparse
import psutil
import os
from pathlib import Path
from typing import List, Tuple, Optional

class DistributedLauncher:
    def __init__(self, shared_dir="./distributed_training", env="HalfCheetah-v4"):
        """
        Smart launcher for distributed RL training
        
        Args:
            shared_dir: Directory for shared model files
            env: Environment name to train on
        """
        self.shared_dir = Path(shared_dir)
        self.env = env
        self.processes = []
        self.coordinator_process = None
        
        # GPU and memory detection
        self.device = self.detect_best_device()
        self.available_memory_gb = self.get_available_gpu_memory()
        self.optimal_workers = self.calculate_optimal_workers()
        
        print(f"🚀 Smart Distributed RL Launcher")
        print(f"Device: {self.device}")
        print(f"Available GPU Memory: {self.available_memory_gb:.1f} GB")
        print(f"Optimal Workers: {self.optimal_workers}")
        print(f"Environment: {self.env}")
        print(f"Shared Directory: {self.shared_dir}")

    def detect_best_device(self) -> str:
        """Detect the best available device"""
        if torch.cuda.is_available():
            # Get the GPU with most memory
            best_gpu = 0
            max_memory = 0
            
            for i in range(torch.cuda.device_count()):
                memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)  # GB
                print(f"GPU {i}: {torch.cuda.get_device_name(i)} - {memory:.1f} GB")
                
                if memory > max_memory:
                    max_memory = memory
                    best_gpu = i
            
            return f"cuda:{best_gpu}"
        else:
            print("⚠️  No CUDA GPUs detected, using CPU")
            return "cpu"

    def get_available_gpu_memory(self) -> float:
        """Get available GPU memory in GB"""
        if "cuda" not in self.device:
            # For CPU, use system RAM
            available_ram = psutil.virtual_memory().available / (1024**3)
            return min(available_ram * 0.8, 32)  # Cap at 32GB, use 80% of available
        
        try:
            gpu_id = int(self.device.split(':')[1])
            torch.cuda.set_device(gpu_id)
            
            # Get total and allocated memory
            total_memory = torch.cuda.get_device_properties(gpu_id).total_memory
            allocated_memory = torch.cuda.memory_allocated(gpu_id)
            reserved_memory = torch.cuda.memory_reserved(gpu_id)
            
            # Calculate available memory (with safety margin)
            available_memory = total_memory - max(allocated_memory, reserved_memory)
            available_gb = available_memory / (1024**3)
            
            # Apply safety margin (85% of available)
            safe_available = available_gb * 0.85
            
            print(f"GPU Memory Analysis:")
            print(f"  Total: {total_memory / (1024**3):.1f} GB")
            print(f"  Allocated: {allocated_memory / (1024**3):.1f} GB")
            print(f"  Reserved: {reserved_memory / (1024**3):.1f} GB")
            print(f"  Available (safe): {safe_available:.1f} GB")
            
            return safe_available
            
        except Exception as e:
            print(f"Error detecting GPU memory: {e}")
            return 4.0  # Conservative fallback

    def calculate_optimal_workers(self) -> int:
        """Calculate optimal number of workers based on available memory"""
        if "cpu" in self.device:
            # For CPU, base on number of cores
            cpu_cores = mp.cpu_count()
            return min(cpu_cores // 2, 8)  # Use half the cores, max 8
        
        # GPU memory requirements (empirical estimates for PPO)
        memory_per_worker = {
            # Memory per worker in GB (training + inference)
            "small_model": 1.2,   # Small networks like your setup
            "medium_model": 2.0,  # Medium networks  
            "large_model": 3.5,   # Large networks
        }
        
        # Coordinator overhead
        coordinator_overhead = 0.5  # GB
        
        # Calculate workers based on model size
        model_type = self.estimate_model_size()
        memory_per_worker_gb = memory_per_worker[model_type]
        
        available_for_workers = self.available_memory_gb - coordinator_overhead
        max_workers = int(available_for_workers / memory_per_worker_gb)
        
        # Apply practical constraints
        optimal_workers = min(max_workers, 12)  # Cap at 12 workers
        optimal_workers = max(optimal_workers, 1)  # At least 1 worker
        
        print(f"Memory Analysis:")
        print(f"  Model type: {model_type}")
        print(f"  Memory per worker: {memory_per_worker_gb:.1f} GB")
        print(f"  Coordinator overhead: {coordinator_overhead:.1f} GB")
        print(f"  Available for workers: {available_for_workers:.1f} GB")
        print(f"  Calculated optimal workers: {optimal_workers}")
        
        return optimal_workers

    def estimate_model_size(self) -> str:
        """Estimate model size based on environment"""
        # Simple heuristic based on environment complexity
        complex_envs = ["Humanoid", "Ant", "Walker2d"]
        medium_envs = ["HalfCheetah", "Hopper", "Swimmer"]
        
        env_name = self.env.lower()
        
        if any(complex_env.lower() in env_name for complex_env in complex_envs):
            return "large_model"
        elif any(medium_env.lower() in env_name for medium_env in medium_envs):
            return "medium_model"
        else:
            return "small_model"

    def create_worker_command(self, worker_id: int, **kwargs) -> List[str]:
        """Create command to launch a worker"""
        cmd = [
            sys.executable, "distributed_worker.py",
            "--worker_id", str(worker_id),
            "--shared_dir", str(self.shared_dir),
            "--env", self.env,
            "--device", self.device
        ]
        
        # Add optional arguments
        for key, value in kwargs.items():
            if value is not None:
                cmd.extend([f"--{key}", str(value)])
        
        return cmd

    def create_coordinator_command(self, **kwargs) -> List[str]:
        """Create command to launch coordinator"""
        cmd = [
            sys.executable, "distributed_coordinator.py",
            "--shared_dir", str(self.shared_dir),
            "--num_workers", str(self.optimal_workers)
        ]
        
        # Add optional arguments
        for key, value in kwargs.items():
            if value is not None:
                cmd.extend([f"--{key}", str(value)])
        
        return cmd

    def start_coordinator(self, **coordinator_kwargs):
        """Start the coordinator process"""
        print(f"🎯 Starting coordinator...")
        
        try:
            cmd = self.create_coordinator_command(**coordinator_kwargs)
            self.coordinator_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            print(f"✅ Coordinator started (PID: {self.coordinator_process.pid})")
            return True
            
        except Exception as e:
            print(f"❌ Failed to start coordinator: {e}")
            return False

    def start_workers(self, **worker_kwargs):
        """Start all worker processes"""
        print(f"👥 Starting {self.optimal_workers} workers...")
        
        for worker_id in range(self.optimal_workers):
            try:
                cmd = self.create_worker_command(worker_id, **worker_kwargs)
                
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    universal_newlines=True,
                    bufsize=1
                )
                
                self.processes.append({
                    'id': worker_id,
                    'process': process,
                    'cmd': cmd
                })
                
                print(f"✅ Worker {worker_id} started (PID: {process.pid})")
                time.sleep(1)  # Stagger startup to avoid conflicts
                
            except Exception as e:
                print(f"❌ Failed to start worker {worker_id}: {e}")

    def monitor_processes(self, show_output=False):
        """Monitor all processes and handle failures"""
        print(f"👀 Monitoring {len(self.processes)} workers + coordinator...")
        
        try:
            while True:
                # Check coordinator
                if self.coordinator_process and self.coordinator_process.poll() is not None:
                    print(f"⚠️  Coordinator process exited (code: {self.coordinator_process.returncode})")
                    break
                
                # Check workers
                for worker_info in self.processes:
                    process = worker_info['process']
                    worker_id = worker_info['id']
                    
                    if process.poll() is not None:
                        print(f"⚠️  Worker {worker_id} exited (code: {process.returncode})")
                        # Optionally restart failed workers
                        # self.restart_worker(worker_info)
                
                # Show output if requested
                if show_output:
                    self.display_recent_output()
                
                time.sleep(5)  # Check every 5 seconds
                
        except KeyboardInterrupt:
            print(f"\n🛑 Monitoring interrupted")

    def display_recent_output(self):
        """Display recent output from processes"""
        # This is a simplified version - you could make it more sophisticated
        if self.coordinator_process:
            try:
                # Non-blocking read from coordinator
                line = self.coordinator_process.stdout.readline()
                if line.strip():
                    print(f"[COORD] {line.strip()}")
            except:
                pass

    def restart_worker(self, worker_info):
        """Restart a failed worker"""
        worker_id = worker_info['id']
        print(f"🔄 Restarting worker {worker_id}...")
        
        try:
            # Start new process
            new_process = subprocess.Popen(
                worker_info['cmd'],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            # Update process info
            worker_info['process'] = new_process
            print(f"✅ Worker {worker_id} restarted (PID: {new_process.pid})")
            
        except Exception as e:
            print(f"❌ Failed to restart worker {worker_id}: {e}")

    def cleanup(self):
        """Clean shutdown of all processes"""
        print(f"\n🧹 Cleaning up processes...")
        
        # Terminate workers
        for worker_info in self.processes:
            process = worker_info['process']
            worker_id = worker_info['id']
            
            try:
                if process.poll() is None:  # Process still running
                    print(f"Terminating worker {worker_id}...")
                    process.terminate()
                    
                    # Wait a bit for graceful shutdown
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        print(f"Force killing worker {worker_id}...")
                        process.kill()
                        
            except Exception as e:
                print(f"Error terminating worker {worker_id}: {e}")
        
        # Terminate coordinator
        if self.coordinator_process:
            try:
                if self.coordinator_process.poll() is None:
                    print(f"Terminating coordinator...")
                    self.coordinator_process.terminate()
                    
                    try:
                        self.coordinator_process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        print(f"Force killing coordinator...")
                        self.coordinator_process.kill()
                        
            except Exception as e:
                print(f"Error terminating coordinator: {e}")
        
        print(f"✅ Cleanup complete")

    def run(self, coordinator_kwargs=None, worker_kwargs=None, show_output=False):
        """Run the complete distributed training setup"""
        coordinator_kwargs = coordinator_kwargs or {}
        worker_kwargs = worker_kwargs or {}
        
        # Setup signal handlers for graceful shutdown
        def signal_handler(signum, frame):
            print(f"\n🛑 Received signal {signum}")
            self.cleanup()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        try:
            # Create shared directory
            self.shared_dir.mkdir(parents=True, exist_ok=True)
            
            # Start coordinator
            if not self.start_coordinator(**coordinator_kwargs):
                return False
            
            # Wait a bit for coordinator to initialize
            time.sleep(3)
            
            # Start workers
            self.start_workers(**worker_kwargs)
            
            # Wait a bit for workers to initialize
            time.sleep(5)
            
            print(f"🚀 Distributed training launched successfully!")
            print(f"📊 {self.optimal_workers} workers training on {self.device}")
            print(f"📁 Shared directory: {self.shared_dir}")
            print(f"🎮 Environment: {self.env}")
            print(f"\n Press Ctrl+C to stop training gracefully")
            
            # Monitor processes
            self.monitor_processes(show_output=show_output)
            
        except Exception as e:
            print(f"💥 Error during execution: {e}")
        finally:
            self.cleanup()

def main():
    parser = argparse.ArgumentParser(description='Smart Distributed RL Launcher')
    
    # Basic settings
    parser.add_argument('--shared_dir', type=str, default='./distributed_training',
                       help='Shared directory for model exchange')
    parser.add_argument('--env', type=str, default='HalfCheetah-v4',
                       help='Environment name')
    parser.add_argument('--show_output', action='store_true',
                       help='Show process output during monitoring')
    
    # Coordinator settings
    parser.add_argument('--check_interval', type=int, default=60,
                       help='Coordinator check interval in seconds')
    parser.add_argument('--max_runtime_hours', type=float, default=None,
                       help='Maximum runtime in hours')
    
    # Worker settings  
    parser.add_argument('--max_episodes', type=int, default=100,
                       help='Episodes per worker checkpoint')
    parser.add_argument('--lr', type=float, default=3e-4,
                       help='Learning rate')
    parser.add_argument('--total_frames', type=int, default=1000000,
                       help='Total training frames per worker')
    
    args = parser.parse_args()
    
    # Separate coordinator and worker arguments
    coordinator_kwargs = {
        'check_interval': args.check_interval,
        'max_runtime_hours': args.max_runtime_hours
    }
    
    worker_kwargs = {
        'max_episodes': args.max_episodes,
        'lr': args.lr,
        'total_frames': args.total_frames
    }
    
    # Create and run launcher
    launcher = DistributedLauncher(
        shared_dir=args.shared_dir,
        env=args.env
    )
    
    launcher.run(
        coordinator_kwargs=coordinator_kwargs,
        worker_kwargs=worker_kwargs,
        show_output=args.show_output
    )

if __name__ == "__main__":
    main()