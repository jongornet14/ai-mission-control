#!/usr/bin/env python3
"""
Coordinator Entry Script for Distributed Training
Uses the MinimalCoordinator class
"""

import sys
import argparse
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, '/workspace/project')

# Import your coordinator
from intellinaut.workers.coordinator import MinimalCoordinator


class EnhancedCoordinator(MinimalCoordinator):
    """
    Enhanced coordinator with better logging and status tracking
    """
    
    def __init__(self, shared_dir, num_workers=4, check_interval=30):
        super().__init__(shared_dir, num_workers, check_interval)
        
        # Create status file for Docker monitoring
        self.status_file = self.shared_dir / "coordinator_status.json"
        self.start_time = __import__('time').time()
        self.sync_count = 0
        
        self._update_status("INITIALIZING")
    
    def _update_status(self, status, **kwargs):
        """Update coordinator status for Docker monitoring"""
        import json
        import time
        from datetime import datetime
        
        status_data = {
            'status': status,
            'sync_count': self.sync_count,
            'uptime_seconds': int(time.time() - self.start_time),
            'last_heartbeat': datetime.now().isoformat(),
            'num_workers': self.num_workers,
            **kwargs
        }
        
        try:
            with open(self.status_file, 'w') as f:
                json.dump(status_data, f, indent=2)
        except Exception as e:
            print(f"Coordinator: Status write error: {e}")
        
        # Also log to stdout for Docker logs
        print(f"COORDINATOR_STATUS: {status} sync_count={self.sync_count}")
    
    def run(self):
        """Enhanced coordination loop with status tracking"""
        print(f"🚀 Coordinator: Starting coordination (check every {self.check_interval}s)")
        self._update_status("RUNNING")
        
        try:
            while not self.should_terminate():
                self.sync_count += 1
                self._update_status("SYNCING", current_sync=self.sync_count)
                
                print(f"\n🔄 Sync #{self.sync_count}")
                
                # Find best worker
                best_worker = self.find_best_worker()
                
                if best_worker is not None:
                    # Copy their model
                    if self.copy_best_model(best_worker):
                        # Signal other workers to update
                        signaled = self.signal_workers(exclude_worker=best_worker)
                        self._update_status(
                            "SYNC_COMPLETE", 
                            best_worker=best_worker,
                            signaled_workers=signaled
                        )
                        print(f"✅ Sync #{self.sync_count} complete - Best: Worker {best_worker}")
                    else:
                        self._update_status("SYNC_FAILED", reason="model_copy_failed")
                        print(f"❌ Sync #{self.sync_count} failed - Model copy error")
                else:
                    self._update_status("SYNC_WAITING", reason="no_eligible_workers")
                    print(f"⏳ Sync #{self.sync_count} - Waiting for eligible workers")
                
                # Wait before next check
                self._update_status("SLEEPING")
                print(f"😴 Sleeping {self.check_interval}s...")
                __import__('time').sleep(self.check_interval)
                
        except KeyboardInterrupt:
            self._update_status("INTERRUPTED")
            print("\n🛑 Coordinator stopped by user")
        except Exception as e:
            self._update_status("ERROR", error=str(e))
            print(f"💥 Coordinator error: {e}")
            raise
        finally:
            self._update_status("SHUTDOWN")
            print(f"📊 Coordinator finished after {self.sync_count} syncs")


def main():
    """Main entry point for coordinator"""
    parser = argparse.ArgumentParser(description='Enhanced Distributed Coordinator')
    
    parser.add_argument('--shared_dir', required=True, help='Shared directory path')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers')
    parser.add_argument('--check_interval', type=int, default=30, help='Seconds between checks')
    
    args = parser.parse_args()
    
    print(f"🎯 Starting Coordinator")
    print(f"📁 Shared directory: {args.shared_dir}")
    print(f"👥 Managing {args.num_workers} workers")
    print(f"⏱️  Check interval: {args.check_interval}s")
    
    # Create enhanced coordinator
    coordinator = EnhancedCoordinator(
        shared_dir=args.shared_dir,
        num_workers=args.num_workers,
        check_interval=args.check_interval
    )
    
    # Run coordination
    coordinator.run()


if __name__ == "__main__":
    main()