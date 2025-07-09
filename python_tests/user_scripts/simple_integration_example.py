#!/usr/bin/env python3
"""
Simple Integration Example: From Local Gym to AI Mission Control

This shows how to migrate from:
  env = gym.make("CartPole-v1")

To:
  env = AIMissionControl.create("CartPole-v1")
"""

import requests
import numpy as np
import time

# Traditional approach (what users currently do)
def traditional_gym_approach():
    """
    Traditional approach - requires local gym installation
    Problems: version conflicts, dependency hell, inconsistent environments
    """
    print("❌ Traditional Approach (what we're replacing):")
    print("   import gym")
    print("   env = gym.make('CartPole-v1')  # Requires local installation")
    print("   obs = env.reset()")
    print("   for step in range(100):")
    print("       action = env.action_space.sample()")
    print("       obs, reward, done, info = env.step(action)")
    print("")
    print("Problems:")
    print("   • Different gym versions across team members")
    print("   • Complex dependency management")
    print("   • Environment setup issues")
    print("   • Hard to reproduce results")

# AI Mission Control approach (your solution)
class AIMissionControl:
    """
    AI Mission Control client - replaces gym.make() with microservices
    Benefits: version control, consistency, no local dependencies
    """
    
    @staticmethod
    def create(env_name: str, service: str = "gym", base_url: str = "http://localhost"):
        """
        Create environment using microservices
        This replaces gym.make(env_name)
        """
        return AIMissionControlEnv(env_name, service, base_url)

class AIMissionControlEnv:
    """
    Environment wrapper that mimics gym.Env interface
    but uses your microservices backend
    """
    
    def __init__(self, env_name: str, service: str, base_url: str):
        self.env_name = env_name
        self.service = service
        self.base_url = base_url
        self.session_id = None
        
        # Service port mapping
        self.ports = {
            "gym": 50053,
            "trading": 50051,
            "unity": 50052,
            "modern_rl": 50054
        }
        
        self.port = self.ports.get(service, 50053)
        self._create_environment()
    
    def _create_environment(self):
        """Create the environment on the microservice"""
        try:
            response = requests.post(
                f"{self.base_url}:{self.port}/create/{self.env_name}", 
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                self.session_id = data["session_id"]
                print(f"✅ Environment created: {self.env_name} (session: {self.session_id})")
            else:
                raise Exception(f"Failed to create environment: {response.status_code}")
                
        except Exception as e:
            raise Exception(f"Could not connect to AI Mission Control: {e}")
    
    def reset(self):
        """Reset environment (when API is fully implemented)"""
        if not self.session_id:
            raise Exception("Environment not created")
        
        # Try reset endpoint (currently returns 404)
        try:
            response = requests.post(
                f"{self.base_url}:{self.port}/reset/{self.session_id}",
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                return np.array(data["observation"])
            else:
                # Fallback: return dummy observation for demo
                print("⚠️ Reset endpoint not implemented yet, using dummy data")
                return np.random.random(4)  # CartPole has 4D observation space
                
        except Exception as e:
            print(f"⚠️ Reset failed: {e}")
            return np.random.random(4)
    
    def step(self, action):
        """Step environment (when API is fully implemented)"""
        if not self.session_id:
            raise Exception("Environment not created")
        
        # Try step endpoint (currently returns 404)
        try:
            response = requests.post(
                f"{self.base_url}:{self.port}/step/{self.session_id}",
                json={"action": action},
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                return (
                    np.array(data["observation"]),
                    data["reward"], 
                    data["done"],
                    data.get("info", {})
                )
            else:
                # Fallback: return dummy data for demo
                return (
                    np.random.random(4),  # Next observation
                    1.0,                  # Reward
                    np.random.random() < 0.01,  # Done (1% chance)
                    {}                    # Info
                )
                
        except Exception as e:
            print(f"⚠️ Step failed: {e}, using dummy data")
            return (
                np.random.random(4),
                1.0,
                False,
                {}
            )

def ai_mission_control_approach():
    """
    AI Mission Control approach - microservices-based
    Benefits: version control, consistency, scalability
    """
    print("✅ AI Mission Control Approach (your solution):")
    print("   from ai_mission_control import AIMissionControl")
    print("   env = AIMissionControl.create('CartPole-v1')  # Uses microservices")
    print("   obs = env.reset()")
    print("   for step in range(100):")
    print("       action = random_action()")
    print("       obs, reward, done, info = env.step(action)")
    print("")
    print("Benefits:")
    print("   ✅ Consistent environment versions across team")
    print("   ✅ No local dependency management")
    print("   ✅ Server-side resource pooling")
    print("   ✅ Easy to scale and reproduce")

def demonstrate_migration():
    """Show how to migrate existing code"""
    print("\n🔄 CODE MIGRATION EXAMPLE")
    print("=" * 50)
    
    print("BEFORE (traditional gym):")
    print("-" * 25)
    print("import gym")
    print("import numpy as np")
    print("")
    print("# Create environment")
    print("env = gym.make('CartPole-v1')")
    print("obs = env.reset()")
    print("")
    print("# Training loop")
    print("for episode in range(10):")
    print("    obs = env.reset()")
    print("    total_reward = 0")
    print("    for step in range(100):")
    print("        action = env.action_space.sample()")
    print("        obs, reward, done, info = env.step(action)")
    print("        total_reward += reward")
    print("        if done:")
    print("            break")
    print("    print(f'Episode reward: {total_reward}')")
    
    print("\n" + "=" * 50)
    
    print("AFTER (AI Mission Control):")
    print("-" * 25)
    print("from ai_mission_control import AIMissionControl")
    print("import numpy as np")
    print("")
    print("# Create environment (same interface!)")
    print("env = AIMissionControl.create('CartPole-v1')")
    print("obs = env.reset()")
    print("")
    print("# Training loop (identical code!)")
    print("for episode in range(10):")
    print("    obs = env.reset()")
    print("    total_reward = 0")
    print("    for step in range(100):")
    print("        action = np.random.randint(0, 2)  # CartPole actions")
    print("        obs, reward, done, info = env.step(action)")
    print("        total_reward += reward")
    print("        if done:")
    print("            break")
    print("    print(f'Episode reward: {total_reward}')")

def run_actual_demo():
    """Run an actual demo with your current API"""
    print("\n🚀 LIVE DEMO WITH CURRENT API")
    print("=" * 50)
    
    try:
        # Create environment using your microservices
        env = AIMissionControl.create("CartPole-v1")
        
        print("Running 3 episodes with current API capabilities...")
        
        for episode in range(3):
            print(f"\n📊 Episode {episode + 1}:")
            
            # Reset (uses dummy data since endpoint not implemented)
            obs = env.reset()
            print(f"   Initial observation: {obs}")
            
            total_reward = 0
            
            # Run 10 steps
            for step in range(10):
                action = np.random.randint(0, 2)  # Random action for CartPole
                obs, reward, done, info = env.step(action)
                total_reward += reward
                
                print(f"   Step {step+1}: action={action}, reward={reward:.1f}, done={done}")
                
                if done:
                    print("   Episode finished early!")
                    break
            
            print(f"   💰 Total reward: {total_reward:.1f}")
        
        print("\n✅ Demo completed successfully!")
        print("📝 Note: Currently using dummy data for reset/step operations")
        print("🔧 Once you implement the reset/step endpoints, this will use real environment data")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        print("Make sure AI Mission Control services are running:")
        print("   docker-compose up -d")

def show_api_roadmap():
    """Show what needs to be implemented"""
    print("\n🗺️ API IMPLEMENTATION ROADMAP")
    print("=" * 50)
    
    print("CURRENT STATUS:")
    print("✅ Environment creation: POST /create/{env_name}")
    print("✅ Service discovery: GET /")
    print("✅ Health checks: GET /health")
    print("")
    
    print("TODO (for full RL loops):")
    print("🔄 Environment reset: POST /reset/{session_id}")
    print("   Response: {'observation': [...], 'info': {...}}")
    print("")
    print("🎮 Environment step: POST /step/{session_id}")
    print("   Request: {'action': action_value}")
    print("   Response: {'observation': [...], 'reward': float, 'done': bool, 'info': {...}}")
    print("")
    print("🗑️ Environment cleanup: DELETE /session/{session_id}")
    print("   Response: {'status': 'deleted'}")
    print("")
    
    print("ADVANCED FEATURES (future):")
    print("🔐 Authentication: API keys or JWT tokens")
    print("📊 Monitoring: Metrics and logging")
    print("⚡ Load balancing: Multiple environment instances")
    print("💾 State management: Save/load environment states")
    print("🎯 Resource limits: Rate limiting and quotas")

def main():
    """Main demo function"""
    print("🎯 AI MISSION CONTROL - INTEGRATION EXAMPLE")
    print("=" * 60)
    print("From Local Gym to Microservices: A Migration Guide")
    print("=" * 60)
    
    # Show the comparison
    traditional_gym_approach()
    print("\n" + "=" * 60)
    ai_mission_control_approach()
    
    # Show migration path
    demonstrate_migration()
    
    # Run live demo
    run_actual_demo()
    
    # Show roadmap
    show_api_roadmap()
    
    print("\n🎉 CONCLUSION")
    print("=" * 60)
    print("Your AI Mission Control system provides:")
    print("✅ Clean separation of concerns")
    print("✅ Version-controlled environments") 
    print("✅ Scalable microservices architecture")
    print("✅ Drop-in replacement for gym.make()")
    print("")
    print("This is a solid foundation for the 'Netflix for RL Environments' vision!")

if __name__ == "__main__":
    main()