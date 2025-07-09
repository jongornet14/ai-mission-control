#!/usr/bin/env python3
"""
AI Mission Control - Performance and Computation Test Script
Tests computational capabilities, throughput, and performance under load
"""

import asyncio
import aiohttp
import time
import json
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
import threading
import queue
from dataclasses import dataclass
from typing import List, Dict, Optional
import argparse
import sys

@dataclass
class PerformanceMetrics:
    """Store performance metrics"""
    service_name: str
    operation: str
    response_times: List[float]
    success_rate: float
    throughput: float
    errors: List[str]

class AIPerformanceTester:
    """Performance testing class for AI Mission Control"""
    
    def __init__(self, base_url: str = "http://localhost"):
        self.base_url = base_url
        self.services = {
            "gym": {"port": 50053, "name": "Gym Service"},
            "trading": {"port": 50051, "name": "Trading Service"},
            "unity": {"port": 50052, "name": "Unity Service"},
            "modern_rl": {"port": 50054, "name": "Modern RL Service"},
            "gateway": {"port": 8080, "name": "API Gateway"}
        }
        self.results = []
        
    def print_header(self, title: str):
        """Print formatted header"""
        print(f"\n{'='*60}")
        print(f"  {title}")
        print(f"{'='*60}")
    
    def print_metric(self, metric: str, value: str, status: str = "INFO"):
        """Print formatted metric"""
        colors = {
            "PASS": "\033[92m",
            "FAIL": "\033[91m", 
            "WARN": "\033[93m",
            "INFO": "\033[94m"
        }
        reset = "\033[0m"
        print(f"{colors.get(status, '')}{status:>6}{reset} {metric}: {value}")
    
    async def test_service_health_performance(self, service_name: str, num_requests: int = 100):
        """Test service health endpoint performance"""
        service = self.services[service_name]
        url = f"{self.base_url}:{service['port']}/health"
        
        response_times = []
        errors = []
        
        async with aiohttp.ClientSession() as session:
            for i in range(num_requests):
                start_time = time.time()
                try:
                    async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                        await response.json()
                        end_time = time.time()
                        response_times.append((end_time - start_time) * 1000)
                        
                        if response.status != 200:
                            errors.append(f"Request {i}: HTTP {response.status}")
                except Exception as e:
                    errors.append(f"Request {i}: {str(e)}")
                    end_time = time.time()
                    response_times.append((end_time - start_time) * 1000)
        
        success_rate = ((num_requests - len(errors)) / num_requests) * 100
        avg_response_time = np.mean(response_times)
        throughput = num_requests / (sum(response_times) / 1000)
        
        metrics = PerformanceMetrics(
            service_name=service['name'],
            operation="health_check",
            response_times=response_times,
            success_rate=success_rate,
            throughput=throughput,
            errors=errors
        )
        
        self.results.append(metrics)
        return metrics
    
    async def test_environment_creation_performance(self, num_environments: int = 50):
        """Test environment creation performance"""
        gym_url = f"{self.base_url}:50053"
        
        response_times = []
        errors = []
        session_ids = []
        
        async with aiohttp.ClientSession() as session:
            for i in range(num_environments):
                start_time = time.time()
                try:
                    async with session.post(
                        f"{gym_url}/create/CartPole-v1",
                        timeout=aiohttp.ClientTimeout(total=30)
                    ) as response:
                        data = await response.json()
                        end_time = time.time()
                        response_times.append((end_time - start_time) * 1000)
                        
                        if response.status == 200:
                            session_ids.append(data.get("session_id"))
                        else:
                            errors.append(f"Environment {i}: HTTP {response.status}")
                            
                except Exception as e:
                    errors.append(f"Environment {i}: {str(e)}")
                    end_time = time.time()
                    response_times.append((end_time - start_time) * 1000)
        
        success_rate = ((num_environments - len(errors)) / num_environments) * 100
        avg_response_time = np.mean(response_times)
        throughput = num_environments / (sum(response_times) / 1000)
        
        metrics = PerformanceMetrics(
            service_name="Gym Service",
            operation="environment_creation",
            response_times=response_times,
            success_rate=success_rate,
            throughput=throughput,
            errors=errors
        )
        
        self.results.append(metrics)
        return metrics, session_ids
    
    async def test_environment_step_performance(self, session_ids: List[str], steps_per_env: int = 100):
        """Test environment step performance"""
        gym_url = f"{self.base_url}:50053"
        
        response_times = []
        errors = []
        total_steps = 0
        
        async with aiohttp.ClientSession() as session:
            for session_id in session_ids[:10]:  # Test first 10 environments
                # Reset environment first
                try:
                    async with session.post(f"{gym_url}/reset/{session_id}") as response:
                        if response.status != 200:
                            continue
                except:
                    continue
                
                # Run steps
                for step in range(steps_per_env):
                    start_time = time.time()
                    try:
                        action = np.random.randint(0, 2)
                        async with session.post(
                            f"{gym_url}/step/{session_id}",
                            json={"action": action},
                            timeout=aiohttp.ClientTimeout(total=10)
                        ) as response:
                            data = await response.json()
                            end_time = time.time()
                            response_times.append((end_time - start_time) * 1000)
                            total_steps += 1
                            
                            if response.status != 200:
                                errors.append(f"Step {step}: HTTP {response.status}")
                            elif data.get("done", False):
                                break  # Episode finished
                                
                    except Exception as e:
                        errors.append(f"Step {step}: {str(e)}")
                        end_time = time.time()
                        response_times.append((end_time - start_time) * 1000)
        
        success_rate = ((total_steps - len(errors)) / total_steps) * 100 if total_steps > 0 else 0
        avg_response_time = np.mean(response_times) if response_times else 0
        throughput = total_steps / (sum(response_times) / 1000) if response_times else 0
        
        metrics = PerformanceMetrics(
            service_name="Gym Service",
            operation="environment_step",
            response_times=response_times,
            success_rate=success_rate,
            throughput=throughput,
            errors=errors
        )
        
        self.results.append(metrics)
        return metrics
    
    def test_concurrent_environment_usage(self, num_threads: int = 10, steps_per_thread: int = 100):
        """Test concurrent environment usage"""
        gym_url = f"{self.base_url}:50053"
        
        def worker_thread(thread_id: int, results_queue: queue.Queue):
            """Worker thread for concurrent testing"""
            import requests
            
            thread_results = {
                "thread_id": thread_id,
                "response_times": [],
                "errors": [],
                "total_steps": 0
            }
            
            try:
                # Create environment
                response = requests.post(f"{gym_url}/create/CartPole-v1", timeout=30)
                if response.status_code != 200:
                    thread_results["errors"].append("Environment creation failed")
                    results_queue.put(thread_results)
                    return
                
                session_id = response.json()["session_id"]
                
                # Reset environment
                requests.post(f"{gym_url}/reset/{session_id}", timeout=10)
                
                # Run steps
                for step in range(steps_per_thread):
                    start_time = time.time()
                    try:
                        action = np.random.randint(0, 2)
                        step_response = requests.post(
                            f"{gym_url}/step/{session_id}",
                            json={"action": action},
                            timeout=10
                        )
                        end_time = time.time()
                        
                        thread_results["response_times"].append((end_time - start_time) * 1000)
                        thread_results["total_steps"] += 1
                        
                        if step_response.status_code != 200:
                            thread_results["errors"].append(f"Step {step}: HTTP {step_response.status_code}")
                        elif step_response.json().get("done", False):
                            # Reset and continue
                            requests.post(f"{gym_url}/reset/{session_id}", timeout=10)
                            
                    except Exception as e:
                        thread_results["errors"].append(f"Step {step}: {str(e)}")
                        end_time = time.time()
                        thread_results["response_times"].append((end_time - start_time) * 1000)
                        
            except Exception as e:
                thread_results["errors"].append(f"Thread setup: {str(e)}")
            
            results_queue.put(thread_results)
        
        # Run concurrent threads
        results_queue = queue.Queue()
        threads = []
        
        start_time = time.time()
        for i in range(num_threads):
            thread = threading.Thread(target=worker_thread, args=(i, results_queue))
            thread.start()
            threads.append(thread)
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        end_time = time.time()
        
        # Collect results
        all_response_times = []
        all_errors = []
        total_steps = 0
        
        for _ in range(num_threads):
            thread_result = results_queue.get()
            all_response_times.extend(thread_result["response_times"])
            all_errors.extend(thread_result["errors"])
            total_steps += thread_result["total_steps"]
        
        success_rate = ((total_steps - len(all_errors)) / total_steps) * 100 if total_steps > 0 else 0
        avg_response_time = np.mean(all_response_times) if all_response_times else 0
        throughput = total_steps / (end_time - start_time)
        
        metrics = PerformanceMetrics(
            service_name="Gym Service",
            operation="concurrent_usage",
            response_times=all_response_times,
            success_rate=success_rate,
            throughput=throughput,
            errors=all_errors
        )
        
        self.results.append(metrics)
        return metrics
    
    def test_stress_performance(self, duration_seconds: int = 60):
        """Stress test the system for a given duration"""
        gym_url = f"{self.base_url}:50053"
        
        import requests
        
        response_times = []
        errors = []
        operations_count = 0
        
        # Create a few environments to cycle through
        session_ids = []
        for i in range(5):
            try:
                response = requests.post(f"{gym_url}/create/CartPole-v1", timeout=30)
                if response.status_code == 200:
                    session_ids.append(response.json()["session_id"])
            except:
                pass
        
        if not session_ids:
            return PerformanceMetrics(
                service_name="Gym Service",
                operation="stress_test",
                response_times=[],
                success_rate=0.0,
                throughput=0.0,
                errors=["No environments could be created"]
            )
        
        start_time = time.time()
        end_time = start_time + duration_seconds
        
        while time.time() < end_time:
            session_id = np.random.choice(session_ids)
            
            # Random operation: reset or step
            if np.random.random() < 0.1:  # 10% resets
                operation_start = time.time()
                try:
                    response = requests.post(f"{gym_url}/reset/{session_id}", timeout=5)
                    operation_end = time.time()
                    response_times.append((operation_end - operation_start) * 1000)
                    operations_count += 1
                    
                    if response.status_code != 200:
                        errors.append(f"Reset failed: HTTP {response.status_code}")
                except Exception as e:
                    errors.append(f"Reset error: {str(e)}")
                    operation_end = time.time()
                    response_times.append((operation_end - operation_start) * 1000)
            else:  # 90% steps
                operation_start = time.time()
                try:
                    action = np.random.randint(0, 2)
                    response = requests.post(
                        f"{gym_url}/step/{session_id}",
                        json={"action": action},
                        timeout=5
                    )
                    operation_end = time.time()
                    response_times.append((operation_end - operation_start) * 1000)
                    operations_count += 1
                    
                    if response.status_code != 200:
                        errors.append(f"Step failed: HTTP {response.status_code}")
                except Exception as e:
                    errors.append(f"Step error: {str(e)}")
                    operation_end = time.time()
                    response_times.append((operation_end - operation_start) * 1000)
        
        success_rate = ((operations_count - len(errors)) / operations_count) * 100 if operations_count > 0 else 0
        avg_response_time = np.mean(response_times) if response_times else 0
        throughput = operations_count / duration_seconds
        
        metrics = PerformanceMetrics(
            service_name="Gym Service",
            operation="stress_test",
            response_times=response_times,
            success_rate=success_rate,
            throughput=throughput,
            errors=errors
        )
        
        self.results.append(metrics)
        return metrics
    
    def generate_performance_report(self):
        """Generate a comprehensive performance report"""
        self.print_header("AI Mission Control Performance Report")
        
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r.success_rate >= 95.0)
        
        self.print_metric("Total Tests", str(total_tests))
        self.print_metric("Passed Tests", f"{passed_tests}/{total_tests}", 
                         "PASS" if passed_tests == total_tests else "WARN")
        
        print(f"\n{'Operation':<25} {'Service':<15} {'Avg Time':<12} {'Success':<10} {'Throughput':<12} {'Status'}")
        print("-" * 85)
        
        for result in self.results:
            avg_time = f"{np.mean(result.response_times):.2f}ms" if result.response_times else "N/A"
            success = f"{result.success_rate:.1f}%"
            throughput = f"{result.throughput:.1f}/s"
            
            status = "PASS" if result.success_rate >= 95.0 and np.mean(result.response_times) < 1000 else "FAIL"
            
            print(f"{result.operation:<25} {result.service_name:<15} {avg_time:<12} {success:<10} {throughput:<12} {status}")
        
        # Performance recommendations
        print(f"\n{'Performance Analysis':^60}")
        print("-" * 60)
        
        slow_operations = [r for r in self.results if r.response_times and np.mean(r.response_times) > 500]
        if slow_operations:
            print("⚠️  Slow Operations Detected:")
            for op in slow_operations:
                print(f"   • {op.operation} ({op.service_name}): {np.mean(op.response_times):.2f}ms")
        
        high_error_ops = [r for r in self.results if r.success_rate < 95.0]
        if high_error_ops:
            print("❌ High Error Rate Operations:")
            for op in high_error_ops:
                print(f"   • {op.operation} ({op.service_name}): {op.success_rate:.1f}% success")
        
        if not slow_operations and not high_error_ops:
            print("✅ All operations performing within acceptable limits!")
    
    def plot_performance_charts(self, save_plots: bool = False):
        """Generate performance visualization charts"""
        try:
            import matplotlib.pyplot as plt
            
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('AI Mission Control Performance Metrics', fontsize=16)
            
            # Response time distribution
            all_times = []
            labels = []
            for result in self.results:
                if result.response_times:
                    all_times.extend(result.response_times)
                    labels.extend([result.operation] * len(result.response_times))
            
            if all_times:
                ax1.hist(all_times, bins=50, alpha=0.7, edgecolor='black')
                ax1.set_xlabel('Response Time (ms)')
                ax1.set_ylabel('Frequency')
                ax1.set_title('Response Time Distribution')
                ax1.grid(True, alpha=0.3)
            
            # Success rates
            operations = [r.operation for r in self.results]
            success_rates = [r.success_rate for r in self.results]
            
            bars = ax2.bar(range(len(operations)), success_rates, 
                          color=['green' if sr >= 95 else 'red' for sr in success_rates])
            ax2.set_xlabel('Operations')
            ax2.set_ylabel('Success Rate (%)')
            ax2.set_title('Success Rate by Operation')
            ax2.set_xticks(range(len(operations)))
            ax2.set_xticklabels(operations, rotation=45, ha='right')
            ax2.set_ylim(0, 100)
            ax2.grid(True, alpha=0.3)
            
            # Throughput comparison
            throughputs = [r.throughput for r in self.results]
            ax3.bar(range(len(operations)), throughputs, color='blue', alpha=0.7)
            ax3.set_xlabel('Operations')
            ax3.set_ylabel('Throughput (ops/sec)')
            ax3.set_title('Throughput by Operation')
            ax3.set_xticks(range(len(operations)))
            ax3.set_xticklabels(operations, rotation=45, ha='right')
            ax3.grid(True, alpha=0.3)
            
            # Average response times
            avg_times = [np.mean(r.response_times) if r.response_times else 0 for r in self.results]
            bars = ax4.bar(range(len(operations)), avg_times,
                          color=['green' if t < 500 else 'orange' if t < 1000 else 'red' for t in avg_times])
            ax4.set_xlabel('Operations')
            ax4.set_ylabel('Average Response Time (ms)')
            ax4.set_title('Average Response Time by Operation')
            ax4.set_xticks(range(len(operations)))
            ax4.set_xticklabels(operations, rotation=45, ha='right')
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_plots:
                plt.savefig('ai_mission_control_performance.png', dpi=300, bbox_inches='tight')
                print("📊 Performance charts saved to 'ai_mission_control_performance.png'")
            
            plt.show()
            
        except ImportError:
            print("📊 Matplotlib not available - skipping chart generation")
    
    async def run_comprehensive_performance_tests(self, quick_mode: bool = False):
        """Run all performance tests"""
        self.print_header("Starting Comprehensive Performance Tests")
        
        # Test 1: Service Health Performance
        print("\n🔍 Testing service health performance...")
        for service_name in self.services.keys():
            if service_name == "gateway":
                continue
            try:
                metrics = await self.test_service_health_performance(
                    service_name, 
                    num_requests=50 if quick_mode else 100
                )
                status = "PASS" if metrics.success_rate >= 95.0 else "FAIL"
                self.print_metric(
                    f"{service_name} health", 
                    f"{metrics.success_rate:.1f}% success, {np.mean(metrics.response_times):.2f}ms avg",
                    status
                )
            except Exception as e:
                self.print_metric(f"{service_name} health", f"Error: {str(e)}", "FAIL")
        
        # Test 2: Environment Creation Performance
        print("\n🏗️  Testing environment creation performance...")
        try:
            metrics, session_ids = await self.test_environment_creation_performance(
                num_environments=25 if quick_mode else 50
            )
            status = "PASS" if metrics.success_rate >= 95.0 else "FAIL"
            self.print_metric(
                "Environment creation",
                f"{metrics.success_rate:.1f}% success, {np.mean(metrics.response_times):.2f}ms avg",
                status
            )
        except Exception as e:
            self.print_metric("Environment creation", f"Error: {str(e)}", "FAIL")
            session_ids = []
        
        # Test 3: Environment Step Performance
        if session_ids:
            print("\n🎮 Testing environment step performance...")
            try:
                metrics = await self.test_environment_step_performance(
                    session_ids, 
                    steps_per_env=50 if quick_mode else 100
                )
                status = "PASS" if metrics.success_rate >= 95.0 else "FAIL"
                self.print_metric(
                    "Environment steps",
                    f"{metrics.success_rate:.1f}% success, {np.mean(metrics.response_times):.2f}ms avg",
                    status
                )
            except Exception as e:
                self.print_metric("Environment steps", f"Error: {str(e)}", "FAIL")
        
        # Test 4: Concurrent Usage
        print("\n🔄 Testing concurrent environment usage...")
        try:
            metrics = self.test_concurrent_environment_usage(
                num_threads=5 if quick_mode else 10,
                steps_per_thread=50 if quick_mode else 100
            )
            status = "PASS" if metrics.success_rate >= 90.0 else "FAIL"  # Lower threshold for concurrent
            self.print_metric(
                "Concurrent usage",
                f"{metrics.success_rate:.1f}% success, {metrics.throughput:.1f} ops/sec",
                status
            )
        except Exception as e:
            self.print_metric("Concurrent usage", f"Error: {str(e)}", "FAIL")
        
        # Test 5: Stress Test
        if not quick_mode:
            print("\n🔥 Running stress test...")
            try:
                metrics = self.test_stress_performance(duration_seconds=30)
                status = "PASS" if metrics.success_rate >= 90.0 else "FAIL"
                self.print_metric(
                    "Stress test",
                    f"{metrics.success_rate:.1f}% success, {metrics.throughput:.1f} ops/sec",
                    status
                )
            except Exception as e:
                self.print_metric("Stress test", f"Error: {str(e)}", "FAIL")

def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(description="AI Mission Control Performance Testing")
    parser.add_argument("--quick", action="store_true", help="Run quick performance tests")
    parser.add_argument("--base-url", default="http://localhost", help="Base URL for services")
    parser.add_argument("--save-plots", action="store_true", help="Save performance plots")
    parser.add_argument("--no-charts", action="store_true", help="Skip chart generation")
    
    args = parser.parse_args()
    
    tester = AIPerformanceTester(base_url=args.base_url)
    
    try:
        # Run async tests
        asyncio.run(tester.run_comprehensive_performance_tests(quick_mode=args.quick))
        
        # Generate report
        tester.generate_performance_report()
        
        # Generate charts
        if not args.no_charts:
            tester.plot_performance_charts(save_plots=args.save_plots)
        
        # Final assessment
        print(f"\n{'Final Assessment':^60}")
        print("=" * 60)
        
        total_tests = len(tester.results)
        passed_tests = sum(1 for r in tester.results if r.success_rate >= 95.0)
        
        if passed_tests == total_tests and total_tests > 0:
            print("🎉 EXCELLENT! Your AI Mission Control system is performing optimally!")
            print("✅ All performance metrics within acceptable limits")
            print("✅ System ready for production workloads")
            print("✅ Microservices architecture working efficiently")
            return 0
        elif passed_tests >= total_tests * 0.8:
            print("⚠️  GOOD: Most performance tests passed with minor issues")
            print("🔧 Consider optimizing slower operations")
            print("📈 System suitable for most workloads")
            return 1
        else:
            print("❌ NEEDS ATTENTION: Significant performance issues detected")
            print("🔧 Please review system configuration and resource allocation")
            print("📊 Check logs for detailed error information")
            return 2
            
    except KeyboardInterrupt:
        print("\n🛑 Performance testing interrupted by user")
        return 3
    except Exception as e:
        print(f"\n❌ Performance testing failed: {str(e)}")
        return 4

if __name__ == "__main__":
    sys.exit(main())