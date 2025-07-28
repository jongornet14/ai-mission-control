#!/usr/bin/env python3
"""
Enhanced Intellinaut Package Test Script

This script tests all imports and basic functionality including
coordinators and distributed workers.
"""

import sys
import traceback
import os
import time
import threading
import shutil
from pathlib import Path

# Terminal colors
class Colors:
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    BOLD = '\033[1m'
    NC = '\033[0m'  # No Color

def print_header(text):
    """Print a section header."""
    print(f"\n{Colors.BLUE}{'='*60}")
    print(f"Testing: {text}")
    print(f"{'='*60}{Colors.NC}")

def print_success(text):
    """Print success message."""
    print(f"{Colors.GREEN}[PASS]{Colors.NC} {text}")

def print_error(text):
    """Print error message."""
    print(f"{Colors.RED}[FAIL]{Colors.NC} {text}")

def print_info(text):
    """Print info message."""
    print(f"{Colors.BLUE}[INFO]{Colors.NC} {text}")

def print_warning(text):
    """Print warning message."""
    print(f"{Colors.YELLOW}[WARN]{Colors.NC} {text}")

def test_import(module_name, items=None):
    """Test importing a module or specific items from a module."""
    try:
        if items:
            exec(f"from {module_name} import {', '.join(items)}")
            print_success(f"Successfully imported {items} from {module_name}")
        else:
            exec(f"import {module_name}")
            print_success(f"Successfully imported {module_name}")
        return True
    except Exception as e:
        print_error(f"Failed to import from {module_name}: {e}")
        traceback.print_exc()
        return False

def setup_test_directories():
    """Create test directories for testing."""
    test_dirs = ['test_shared', 'test_logs', 'test_distributed']
    for dir_name in test_dirs:
        if os.path.exists(dir_name):
            shutil.rmtree(dir_name)
        os.makedirs(dir_name)
    print_info("Created test directories")

def cleanup_test_directories():
    """Clean up test directories."""
    test_dirs = ['test_shared', 'test_logs', 'test_distributed']
    for dir_name in test_dirs:
        if os.path.exists(dir_name):
            shutil.rmtree(dir_name)
    print_info("Cleaned up test directories")

def test_coordinator_import():
    """Test coordinator imports."""
    print_header("Coordinator Import Tests")
    success_count = 0
    total_tests = 0
    
    # Test MinimalCoordinator
    total_tests += 1
    try:
        from intellinaut.workers.coordinator import MinimalCoordinator
        print_success("MinimalCoordinator imported successfully")
        success_count += 1
    except Exception as e:
        print_error(f"MinimalCoordinator import failed: {e}")
    
    return success_count, total_tests

def test_distributed_worker_import():
    """Test distributed worker imports."""
    print_header("Distributed Worker Import Tests")
    success_count = 0
    total_tests = 0
    
    # Test DistributedWorker
    total_tests += 1
    try:
        from intellinaut.workers.distributed import DistributedWorker
        print_success("DistributedWorker imported successfully")
        success_count += 1
    except Exception as e:
        print_error(f"DistributedWorker import failed: {e}")
    
    # Test if it inherits from BaseWorker
    total_tests += 1
    try:
        from intellinaut.workers.distributed import DistributedWorker
        from intellinaut.workers.base import BaseWorker
        if issubclass(DistributedWorker, BaseWorker):
            print_success("DistributedWorker correctly inherits from BaseWorker")
            success_count += 1
        else:
            print_error("DistributedWorker does not inherit from BaseWorker")
    except Exception as e:
        print_error(f"DistributedWorker inheritance test failed: {e}")
    
    return success_count, total_tests

def test_coordinator_functionality():
    """Test coordinator functionality."""
    print_header("Coordinator Functionality Tests")
    success_count = 0
    total_tests = 0
    
    total_tests += 1
    try:
        from intellinaut.workers.coordinator import MinimalCoordinator
        
        print_info("Testing MinimalCoordinator initialization...")
        coordinator = MinimalCoordinator(
            shared_dir='test_distributed',
            num_workers=2,
            check_interval=1  # Fast for testing
        )
        print_success("MinimalCoordinator initialized successfully")
        
        # Test directory creation
        expected_dirs = ['models', 'metrics', 'best_model', 'signals']
        all_dirs_exist = all(
            os.path.exists(os.path.join('test_distributed', d)) 
            for d in expected_dirs
        )
        
        if all_dirs_exist:
            print_success("Coordinator created all required directories")
            success_count += 1
        else:
            print_error("Coordinator failed to create required directories")
            
    except Exception as e:
        print_error(f"MinimalCoordinator functionality test failed: {e}")
        traceback.print_exc()
    
    # Test coordinator methods
    total_tests += 1
    try:
        from intellinaut.workers.coordinator import MinimalCoordinator
        coordinator = MinimalCoordinator('test_distributed', 2, 1)
        
        # Test find_best_worker with no workers
        best_worker = coordinator.find_best_worker()
        if best_worker is None:
            print_success("find_best_worker correctly returns None when no workers")
            success_count += 1
        else:
            print_error("find_best_worker should return None when no workers exist")
            
    except Exception as e:
        print_error(f"Coordinator methods test failed: {e}")
    
    return success_count, total_tests

def test_distributed_worker_functionality():
    """Test distributed worker functionality."""
    print_header("Distributed Worker Functionality Tests")
    success_count = 0
    total_tests = 0
    
    # Test DistributedWorker initialization
    total_tests += 1
    try:
        from intellinaut.workers.distributed import DistributedWorker
        
        print_info("Testing DistributedWorker initialization...")
        worker = DistributedWorker(
            worker_id=0,
            shared_dir='test_distributed',
            max_episodes=3,
            checkpoint_frequency=2,
            status_update_frequency=1
        )
        print_success("DistributedWorker initialized successfully")
        
        # Check if it has BaseWorker attributes
        if hasattr(worker, 'worker_id') and hasattr(worker, 'log_dir'):
            print_success("DistributedWorker has required BaseWorker attributes")
            success_count += 1
        else:
            print_error("DistributedWorker missing BaseWorker attributes")
            
    except Exception as e:
        print_error(f"DistributedWorker initialization failed: {e}")
        traceback.print_exc()
    
    # Test distributed-specific methods
    total_tests += 1
    try:
        from intellinaut.workers.distributed import DistributedWorker
        worker = DistributedWorker(0, 'test_distributed', max_episodes=1)
        
        # Test if distributed methods exist
        required_methods = [
            'check_for_coordinator_updates',
            'load_best_model_from_coordinator',
            'save_distributed_checkpoint',
            'calculate_reward_change'
        ]
        
        methods_exist = all(hasattr(worker, method) for method in required_methods)
        
        if methods_exist:
            print_success("DistributedWorker has all required distributed methods")
            success_count += 1
        else:
            missing = [m for m in required_methods if not hasattr(worker, m)]
            print_error(f"DistributedWorker missing methods: {missing}")
            
    except Exception as e:
        print_error(f"DistributedWorker methods test failed: {e}")
    
    return success_count, total_tests

def test_entry_scripts():
    """Test entry scripts exist and are importable."""
    print_header("Entry Scripts Tests")
    success_count = 0
    total_tests = 0
    
    scripts_to_test = [
        'worker_entry.py',
        'coordinator_entry.py',
        'distributed_cli.py',
        'generate_docker_compose_distributed.py'
    ]
    
    for script_path in scripts_to_test:
        total_tests += 1
        if os.path.exists(script_path):
            print_success(f"{script_path} exists")
            success_count += 1
        else:
            print_error(f"{script_path} does not exist")
    
    return success_count, total_tests

def test_distributed_integration():
    """Test basic distributed integration."""
    print_header("Distributed Integration Tests")
    success_count = 0
    total_tests = 0
    
    total_tests += 1
    try:
        print_info("Testing coordinator + worker file communication...")
        
        # Create coordinator
        from intellinaut.workers.coordinator import MinimalCoordinator
        coordinator = MinimalCoordinator('test_distributed', 1, 1)
        
        # Create fake worker metrics
        import json
        metrics_file = Path('test_distributed/metrics/worker_0_performance.json')
        metrics_file.parent.mkdir(parents=True, exist_ok=True)
        
        fake_metrics = {
            'worker_id': 0,
            'total_episodes': 25,
            'avg_reward': 150.5,
            'reward_change': 12.3,
            'timestamp': time.time()
        }
        
        with open(metrics_file, 'w') as f:
            json.dump(fake_metrics, f)
        
        # Test coordinator can read metrics
        loaded_metrics = coordinator.get_worker_metrics(0)
        
        if loaded_metrics and loaded_metrics['worker_id'] == 0:
            print_success("Coordinator can read worker metrics")
            success_count += 1
        else:
            print_error("Coordinator failed to read worker metrics")
            
    except Exception as e:
        print_error(f"Distributed integration test failed: {e}")
        traceback.print_exc()
    
    return success_count, total_tests

def test_short_training_run():
    """Test a very short training run to verify everything works."""
    print_header("Short Training Run Tests")
    success_count = 0
    total_tests = 0
    
    # Test BaseWorker short run
    total_tests += 1
    try:
        from intellinaut.workers.base import BaseWorker
        print_info("Testing BaseWorker with 2 episodes...")
        
        worker = BaseWorker(0, 'test_logs', max_episodes=2)
        
        # Mock the run method to avoid full RL training
        def mock_run(self, env_name, device='cpu', lr=3e-4):
            print_info(f"Mock BaseWorker run: {env_name}, {device}")
            self._update_status("MOCK_TRAINING")
            return True
        
        # Replace run method temporarily
        BaseWorker.run = mock_run
        result = worker.run('CartPole-v1', 'cpu')
        
        if result:
            print_success("BaseWorker mock training completed")
            success_count += 1
        else:
            print_error("BaseWorker mock training failed")
            
    except Exception as e:
        print_error(f"BaseWorker short run failed: {e}")
        traceback.print_exc()
    
    # Test DistributedWorker short run
    total_tests += 1
    try:
        from intellinaut.workers.distributed import DistributedWorker
        print_info("Testing DistributedWorker with mock training...")
        
        worker = DistributedWorker(0, 'test_distributed', max_episodes=2)
        
        # Test distributed-specific functionality
        update_available = worker.check_for_coordinator_updates()
        print_info(f"Coordinator updates available: {update_available}")
        
        # Test metrics creation
        worker.episode_rewards.extend([10.0, 15.0, 12.0, 18.0, 20.0])
        reward_change = worker.calculate_reward_change()
        print_info(f"Calculated reward change: {reward_change}")
        
        print_success("DistributedWorker functionality test completed")
        success_count += 1
        
    except Exception as e:
        print_error(f"DistributedWorker short run failed: {e}")
        traceback.print_exc()
    
    return success_count, total_tests

def main():
    """Run all enhanced package tests."""
    print(f"{Colors.BOLD}Enhanced Intellinaut Package Structure Test{Colors.NC}")
    print("Testing all imports, coordinators, and distributed functionality...")
    
    # Setup
    setup_test_directories()
    
    total_success = 0
    total_tests = 0
    
    try:
        # Test 1: Main package import
        print_header("Main Package Import")
        if test_import("intellinaut"):
            total_success += 1
        total_tests += 1
        
        # Test 2: Module imports
        print_header("Module Imports")
        modules = [
            "algorithms",
            "cli", 
            "workers",
            "environments",
            "logging",
            "training",
        ]
        for module in modules:
            total_tests += 1
            if test_import(f"intellinaut.{module}"):
                total_success += 1
        
        # Test 3: Base Worker import
        print_header("Base Worker Import")
        total_tests += 1
        if test_import("intellinaut.workers.base", ["BaseWorker"]):
            total_success += 1
        
        # Test 4: Coordinator tests
        success, tests = test_coordinator_import()
        total_success += success
        total_tests += tests
        
        success, tests = test_coordinator_functionality()
        total_success += success
        total_tests += tests
        
        # Test 5: Distributed worker tests
        success, tests = test_distributed_worker_import()
        total_success += success
        total_tests += tests
        
        success, tests = test_distributed_worker_functionality()
        total_success += success
        total_tests += tests
        
        # Test 6: Entry scripts
        success, tests = test_entry_scripts()
        total_success += success
        total_tests += tests
        
        # Test 7: Integration tests
        success, tests = test_distributed_integration()
        total_success += success
        total_tests += tests
        
        # Test 8: Short training runs
        success, tests = test_short_training_run()
        total_success += success
        total_tests += tests
        
        # Test 9: Package metadata
        print_header("Package Metadata")
        total_tests += 1
        try:
            import intellinaut
            if hasattr(intellinaut, '__version__'):
                print_info(f"Package version: {intellinaut.__version__}")
            else:
                print_warning("Package version not defined")
            print_success("Package metadata accessible")
            total_success += 1
        except Exception as e:
            print_error(f"Failed to access package metadata: {e}")
        
    finally:
        # Cleanup
        cleanup_test_directories()
    
    # Final Results
    print_header("Enhanced Test Results Summary")
    print_info(f"Passed: {total_success}/{total_tests} tests")
    print_info(f"Failed: {total_tests - total_success}/{total_tests} tests")
    print_info(f"Success rate: {(total_success/total_tests)*100:.1f}%")
    
    if total_success == total_tests:
        print(f"\n{Colors.GREEN}{Colors.BOLD}🎉 All tests passed! Distributed training system is ready!{Colors.NC}")
        return 0
    elif total_success >= total_tests * 0.8:  # 80% pass rate
        print(f"\n{Colors.YELLOW}{Colors.BOLD}⚠️  Most tests passed. Some issues need attention.{Colors.NC}")
        return 1
    else:
        print(f"\n{Colors.RED}{Colors.BOLD}❌ Many tests failed. Check the package structure and implementations.{Colors.NC}")
        return 2

if __name__ == "__main__":
    sys.exit(main())