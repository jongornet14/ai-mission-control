#!/usr/bin/env python3
"""
Intellinaut Package Test Script

This script tests all imports and basic functionality to ensure
the package structure is working correctly.
"""

import sys
import traceback
import os

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
        return False

def main():
    """Run all package tests."""
    print(f"{Colors.BOLD}Intellinaut Package Structure Test{Colors.NC}")
    print("Testing all imports and basic functionality...")
    
    success_count = 0
    total_tests = 0
    
    # Test 1: Main package import
    print_header("Main Package Import")
    total_tests += 1
    if test_import("intellinaut"):
        success_count += 1
    
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
            success_count += 1
    
    # Test 3: Worker Tests
    print_header("Worker Tests")
    
    # Create test directories
    if not os.path.exists('test_shared'):  # Fixed typo
        os.makedirs('test_shared')
    if not os.path.exists('test_logs'):
        os.makedirs('test_logs')
    
    # Test BaseWorker
    total_tests += 1
    try:
        from intellinaut.workers.base import BaseWorker
        print_info("Testing BaseWorker with 3 episodes...")
        BaseWorker(0, './test_logs', 3).run('CartPole-v1', 'cuda:0')
        print_success("BaseWorker test completed")
        success_count += 1
    except Exception as e:
        print_error(f"BaseWorker test failed: {e}")
    
    # Test DistributedWorker
    total_tests += 1
    try:
        from intellinaut.workers.distributed import DistributedWorker
        print_info("Testing DistributedWorker with 3 episodes...")
        DistributedWorker(0, './test_shared', max_episodes=3).run('CartPole-v1', 'cuda:0')
        print_success("DistributedWorker test completed")
        success_count += 1
    except Exception as e:
        print_error(f"DistributedWorker test failed: {e}")
    
    # Test AdaptiveWorker
    total_tests += 1
    try:
        from intellinaut.workers.adaptive import AdaptiveWorker
        print_info("Testing AdaptiveWorker with 5 episodes...")
        AdaptiveWorker(0, './test_shared', max_episodes=5, morphing_frequency=3).run('CartPole-v1', 'cuda:0')
        print_success("AdaptiveWorker test completed")
        success_count += 1
    except Exception as e:
        print_error(f"AdaptiveWorker test failed: {e}")

    # Test 4: Package version and metadata
    print_header("Package Metadata")
    total_tests += 1
    try:
        import intellinaut
        if hasattr(intellinaut, '__version__'):
            print_info(f"Package version: {intellinaut.__version__}")
        if hasattr(intellinaut, '__author__'):
            print_info(f"Author: {intellinaut.__author__}")
        print_success("Package metadata accessible")
        success_count += 1
    except Exception as e:
        print_error(f"Failed to access package metadata: {e}")
    
    # Final Results
    print_header("Test Results Summary")
    print_info(f"Passed: {success_count}/{total_tests} tests")
    print_info(f"Failed: {total_tests - success_count}/{total_tests} tests")
    
    if success_count == total_tests:
        print(f"\n{Colors.GREEN}{Colors.BOLD}All tests passed! Package structure is working correctly.{Colors.NC}")
        return 0
    else:
        print(f"\n{Colors.YELLOW}{Colors.BOLD}{total_tests - success_count} tests failed. Check the package structure.{Colors.NC}")
        return 1

if __name__ == "__main__":
    sys.exit(main())