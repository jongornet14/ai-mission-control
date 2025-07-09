#!/usr/bin/env python3
"""
Quick script to discover the actual API endpoints
"""
import requests
import json

def test_gym_endpoints():
    base_url = "http://localhost:50053"
    
    print("🔍 Discovering Gym Service API endpoints...")
    
    # Test root endpoint
    try:
        response = requests.get(f"{base_url}/", timeout=5)
        print(f"GET / -> {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"Response: {json.dumps(data, indent=2)}")
    except Exception as e:
        print(f"GET / -> Error: {e}")
    
    # Test environment creation
    try:
        response = requests.post(f"{base_url}/create/CartPole-v1", timeout=5)
        print(f"POST /create/CartPole-v1 -> {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"Response: {json.dumps(data, indent=2)}")
            session_id = data.get("session_id")
            
            if session_id:
                # Test different possible endpoint patterns
                test_endpoints = [
                    f"/reset/{session_id}",
                    f"/environments/{session_id}/reset",
                    f"/session/{session_id}/reset",
                    f"/{session_id}/reset",
                    f"/step/{session_id}",
                    f"/environments/{session_id}/step",
                    f"/session/{session_id}/step",
                    f"/{session_id}/step"
                ]
                
                for endpoint in test_endpoints:
                    try:
                        if "reset" in endpoint:
                            response = requests.post(f"{base_url}{endpoint}", timeout=5)
                        else:
                            response = requests.post(f"{base_url}{endpoint}", 
                                                   json={"action": 0}, timeout=5)
                        print(f"POST {endpoint} -> {response.status_code}")
                        if response.status_code == 200:
                            print(f"  ✅ Working endpoint found!")
                            break
                    except Exception as e:
                        print(f"POST {endpoint} -> Error: {e}")
        else:
            print(f"Environment creation failed: {response.text}")
    except Exception as e:
        print(f"POST /create/CartPole-v1 -> Error: {e}")

if __name__ == "__main__":
    test_gym_endpoints()