#!/usr/bin/env python3
"""
AI Mission Control - Simple Usage Example
Shows how to use the microservices API for RL training

This example demonstrates the "Netflix for RL Environments" concept:
- Instead of installing gym locally, use microservices
- Version-controlled environments on the server side
- Clean separation between your algorithm and environment management
"""

import requests
import numpy as np
import time
import argparse
from typing import Dict, Any, Optional, Tuple
import json

class AIMissionControlClient:
    """
    Simple client for AI Mission Control microservices
    
    This replaces direct gym.make() calls with API calls to your microservices
    """
    
    def __init__(self, base_url: str = "http://localhost"):
        self.base_url = base_url
        self.gym_port = 50053
        self.trading_port = 50051
        self.unity_port = 50052
        self.modern_rl_port = 50054
        
        self.active_sessions = {}
        
    def list_environments(self, service: str = "gym") -> Dict[str, Any]:
        """List available environments from a service"""
        port_map = {
            "gym": self.gym_port,
            "trading": self.trading_port, 
            "unity": self.unity_port,
            "modern_rl": self.modern_rl_port
        }
        
        port = port_map.get(service, self.gym_port)
        response = requests.get(f"{self.base_url}:{port}/", timeout=10)
        
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Failed to get environments: {response.status_code}")
    
    def create_environment(self, env_name: str, service: str = "gym") -> str:
        """
        Create an environment and return session ID
        
        This is equivalent to gym.make(env_name) but using microservices
        """
        port_map = {
            "gym": self.gym_port,
            "trading": self.trading_port,
            "unity": self.unity_port, 
            "modern_rl": self.modern_rl_port
        }
        
        port = port_map.get(service, self.gym_port)
        response = requests.post(f"{self.base_url}:{port}/create/{env_name}", timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            session_id = data["session_id"]
            self.active_sessions[session_id] = {
                "env_name": env_name,
                "service": service,
                "port": port
            }
            return session_id
        else:
            raise Exception(f"Failed to create environment: {response.status_code}")
    
    def reset_environment(self, session_id: str) -> Optional[np.ndarray]:
        """Reset environment (when implemented)"""
        if session_id not in self.active_sessions:
            raise Exception(f"Session {session_id} not found")
        
        session_info = self.active_sessions[session_id]
        port = session_info["port"]
        
        # Try different possible endpoint patterns
        reset_endpoints = [
            f"/reset/{session_id}",
            f"/env/{session_id}/reset",
            f"/environments/{session_id}/reset"
        ]
        
        for endpoint in reset_endpoints:
            try:
                response = requests.post(f"{self.base_url}:{port}{endpoint}", timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    return np.array(data["observation"])
            except:
                continue
        
        print(f"⚠️ Reset endpoint not yet implemented for {session_info['env_name']}")
        return None
    
    def step_environment(self, session_id: str, action: int) -> Optional[Tuple[np.ndarray, float, bool, Dict]]:
        """Step environment (when implemented)"""
        if session_id not in self.active_sessions:
            raise Exception(f"Session {session_id} not found")
        
        session_info = self.active_sessions[session_id]
        port = session_info["port"]
        
        # Try different possible endpoint patterns
        step_endpoints = [
            f"/step/{session_id}",
            f"/env/{session_id}/step", 
            f"/environments/{session_id}/step"
        ]
        
        for endpoint in step_endpoints:
            try:
                response = requests.post(f"{self.base_url}:{port}{endpoint}", 
                                       json={"action": action}, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    return (
                        np.array(data["observation"]),
                        data["reward"],
                        data["done"],
                        data.get("info", {})
                    )
            except:
                continue
        
        print(f"⚠️ Step endpoint not yet implemented for {session_info['env_name']}")
        return None

def demonstrate_service_discovery():
    """Demonstrate discovering available environments"""
    print("🔍 AI Mission Control - Service Discovery")
    print("=" * 50)
    
    client = AIMissionControlClient()
    
    services = ["gym", "trading", "unity", "modern_rl"]
    
    for service in services:
        try:
            env_info = client.list_environments(service)
            print(f"\n📋 {service.upper()} Service:")
            print(f"   Version: {env_info.get('version', 'unknown')}")
            
            environments = env_info.get('environments', [])
            print(f"   Available environments: {len(environments)}")
            for env in environments:
                print(f"     • {env}")
                
            if service == "gym":
                box2d = env_info.get('box2d_available', False)
                print(f"   Box2D Physics: {'✅ Available' if box2d else '❌ Not available'}")
                
        except Exception as e:
            print(f"   ❌ {service} service not available: {e}")

def demonstrate_environment_creation():
    """Demonstrate creating environments"""
    print("\n\n🏗️ AI Mission Control - Environment Creation")
    print("=" * 50)
    
    client = AIMissionControlClient()
    
    # Test creating different environment types
    test_environments = [
        ("CartPole-v1", "gym"),
        ("LunarLander-v2", "gym"),
        ("AAPL", "trading"),
    ]
    
    created_sessions = []
    
    for env_name, service in test_environments:
        try:
            print(f"\n🎮 Creating {env_name} on {service} service...")
            session_id = client.create_environment(env_name, service)
            print(f"   ✅ Created successfully!")
            print(f"   📋 Session ID: {session_id}")
            created_sessions.append(session_id)
            
        except Exception as e:
            print(f"   ❌ Failed to create {env_name}: {e}")
    
    return created_sessions

def demonstrate_basic_rl_loop():
    """Demonstrate a basic RL training loop (limited by current API)"""
    print("\n\n🤖 AI Mission Control - Basic RL Loop")
    print("=" * 50)
    
    client = AIMissionControlClient()
    
    try:
        # Create environment
        print("Creating CartPole environment...")
        session_id = client.create_environment("CartPole-v1", "gym")
        print(f"✅ Environment created: {session_id}")
        
        # Try to run basic RL loop
        print("\nAttempting basic RL operations...")
        
        # Reset
        observation = client.reset_environment(session_id)
        if observation is not None:
            print(f"✅ Reset successful, observation shape: {observation.shape}")
            
            # Run a few steps
            total_reward = 0
            for step in range(10):
                action = np.random.randint(0, 2)  # Random action for CartPole
                result = client.step_environment(session_id, action)
                
                if result is not None:
                    obs, reward, done, info = result
                    total_reward += reward
                    print(f"   Step {step}: action={action}, reward={reward}, done={done}")
                    
                    if done:
                        print("   Episode finished!")
                        break
                else:
                    print(f"   Step {step}: API not fully implemented yet")
                    break
            
            print(f"✅ Total reward: {total_reward}")
        else:
            print("⚠️ Reset/step endpoints not yet implemented")
            print("📝 Current API supports environment creation only")
            
    except Exception as e:
        print(f"❌ RL loop failed: {e}")

def demonstrate_microservices_benefits():
    """Show the benefits of the microservices approach"""
    print("\n\n🌟 AI Mission Control - Microservices Benefits")
    print("=" * 50)
    
    client = AIMissionControlClient()
    
    print("🎯 Benefits of your 'Netflix for RL Environments' approach:")
    print()
    
    # 1. Version Control
    print("1. 📦 VERSION CONTROL:")
    try:
        gym_info = client.list_environments("gym")
        version = gym_info.get('version', 'unknown')
        print(f"   • Gym service version: {version}")
        print(f"   • Server-side package management")
        print(f"   • No local environment conflicts")
        print(f"   • Consistent environments across team")
    except:
        print("   • Version info not available")
    
    # 2. Multiple Environment Types
    print("\n2. 🎮 MULTIPLE ENVIRONMENT TYPES:")
    services = ["gym", "trading", "unity", "modern_rl"]
    total_envs = 0
    
    for service in services:
        try:
            env_info = client.list_environments(service)
            env_count = len(env_info.get('environments', []))
            total_envs += env_count
            print(f"   • {service.title()}: {env_count} environments")
        except:
            print(f"   • {service.title()}: service not available")
    
    print(f"   • Total: {total_envs} environments across all services")
    
    # 3. Scalability
    print("\n3. ⚡ SCALABILITY:")
    start_time = time.time()
    
    # Create multiple environments quickly
    sessions = []
    for i in range(5):
        try:
            session_id = client.create_environment("CartPole-v1", "gym")
            sessions.append(session_id)
        except:
            pass
    
    end_time = time.time()
    
    print(f"   • Created {len(sessions)} environments in {end_time - start_time:.3f}s")
    print(f"   • Can run multiple algorithms simultaneously")
    print(f"   • Resource pooling on server side")
    
    # 4. Clean APIs
    print("\n4. 🔧 CLEAN API DESIGN:")
    print("   • Simple REST endpoints")
    print("   • Language agnostic (any language can use HTTP)")
    print("   • Easy integration with existing ML pipelines")
    print("   • Microservices can be updated independently")

def simulate_training_workflow():
    """Simulate what a training workflow would look like"""
    print("\n\n🚀 AI Mission Control - Training Workflow Simulation")
    print("=" * 50)
    
    client = AIMissionControlClient()
    
    # Simulated hyperparameter tuning
    hyperparams = [
        {"lr": 0.001, "env": "CartPole-v1"},
        {"lr": 0.003, "env": "CartPole-v1"}, 
        {"lr": 0.01, "env": "LunarLander-v2"},
    ]
    
    print("🧪 Simulating hyperparameter tuning across environments...")
    
    results = []
    
    for i, params in enumerate(hyperparams):
        print(f"\n📊 Experiment {i+1}: lr={params['lr']}, env={params['env']}")
        
        try:
            # Create environment for this experiment
            session_id = client.create_environment(params['env'], "gym")
            print(f"   ✅ Environment ready: {session_id}")
            
            # Simulate training metrics
            simulated_reward = np.random.normal(100, 20)  # Fake reward
            training_time = np.random.uniform(60, 120)    # Fake training time
            
            results.append({
                "experiment": i+1,
                "lr": params['lr'],
                "env": params['env'],
                "final_reward": simulated_reward,
                "training_time": training_time,
                "session_id": session_id
            })
            
            print(f"   📈 Simulated training complete")
            print(f"   🎯 Final reward: {simulated_reward:.2f}")
            print(f"   ⏱️ Training time: {training_time:.1f}s")
            
        except Exception as e:
            print(f"   ❌ Experiment failed: {e}")
    
    # Show results
    print(f"\n📊 Hyperparameter Tuning Results:")
    print("=" * 60)
    print(f"{'Exp':<4} {'LR':<6} {'Environment':<15} {'Reward':<8} {'Time':<6}")
    print("-" * 60)
    
    for result in results:
        print(f"{result['experiment']:<4} {result['lr']:<6} {result['env']:<15} "
              f"{result['final_reward']:<8.1f} {result['training_time']:<6.1f}s")
    
    if results:
        best_result = max(results, key=lambda x: x['final_reward'])
        print(f"\n🏆 Best configuration: lr={best_result['lr']}, env={best_result['env']}")

def main():
    """Main demonstration function"""
    parser = argparse.ArgumentParser(description="AI Mission Control Usage Example")
    parser.add_argument("--demo", choices=["all", "discovery", "creation", "rl", "benefits", "training"],
                       default="all", help="Which demo to run")
    parser.add_argument("--base-url", default="http://localhost", 
                       help="Base URL for AI Mission Control services")
    
    args = parser.parse_args()
    
    print("🎯 AI MISSION CONTROL - USAGE EXAMPLE")
    print("=" * 60)
    print("Welcome to your 'Netflix for RL Environments' system!")
    print("This example shows how to use microservices instead of local gym installs.")
    print("=" * 60)
    
    # Override base URL if provided
    if args.base_url != "http://localhost":
        print(f"🌐 Using custom base URL: {args.base_url}")
    
    try:
        if args.demo in ["all", "discovery"]:
            demonstrate_service_discovery()
        
        if args.demo in ["all", "creation"]:
            demonstrate_environment_creation()
        
        if args.demo in ["all", "rl"]:
            demonstrate_basic_rl_loop()
        
        if args.demo in ["all", "benefits"]:
            demonstrate_microservices_benefits()
        
        if args.demo in ["all", "training"]:
            simulate_training_workflow()
        
        print("\n\n🎉 Demo completed!")
        print("=" * 60)
        print("Next steps:")
        print("1. Implement reset/step endpoints in your services")
        print("2. Add authentication and rate limiting")
        print("3. Implement environment state management")
        print("4. Add monitoring and logging")
        print("5. Create client libraries for different languages")
        print("")
        print("Your microservices architecture is solid! 🚀")
        
    except KeyboardInterrupt:
        print("\n🛑 Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        print("Make sure your AI Mission Control services are running:")
        print("  docker-compose up -d")

if __name__ == "__main__":
    main()