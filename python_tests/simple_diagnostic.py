#!/usr/bin/env python3
"""
Simple diagnostic to see what's actually working
"""
import requests
import json

def main():
    base_url = "http://localhost:50053"
    
    print("🔍 Simple API Diagnostic")
    print("=" * 40)
    
    # 1. Test service info
    print("\n1. Testing service info...")
    try:
        response = requests.get(f"{base_url}/", timeout=5)
        print(f"GET / -> {response.status_code}")
        if response.status_code == 200:
            print(f"Response: {json.dumps(response.json(), indent=2)}")
    except Exception as e:
        print(f"Error: {e}")
    
    # 2. Test environment creation
    print("\n2. Testing environment creation...")
    try:
        response = requests.post(f"{base_url}/create/CartPole-v1", timeout=5)
        print(f"POST /create/CartPole-v1 -> {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"Response: {json.dumps(data, indent=2)}")
            
            session_id = data.get("session_id")
            if session_id:
                print(f"\n3. Session ID: {session_id}")
                
                # Test what endpoints return 404
                test_urls = [
                    f"/reset/{session_id}",
                    f"/step/{session_id}",
                    f"/environments/{session_id}/reset",
                    f"/environments/{session_id}/step"
                ]
                
                for url in test_urls:
                    try:
                        if "step" in url:
                            resp = requests.post(f"{base_url}{url}", json={"action": 0}, timeout=5)
                        else:
                            resp = requests.post(f"{base_url}{url}", timeout=5)
                        print(f"POST {url} -> {resp.status_code}")
                        if resp.status_code != 404:
                            print(f"  Response: {resp.text[:100]}...")
                    except Exception as e:
                        print(f"POST {url} -> Error: {e}")
        else:
            print(f"Failed: {response.text}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()