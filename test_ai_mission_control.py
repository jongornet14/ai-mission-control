"""AI Mission Control - Basic Test Suite"""

import pytest
import requests
import json

SERVICES = {
    "trading": {"port": 50051, "name": "Trading Service"},
    "unity": {"port": 50052, "name": "Unity Service"},
    "gym": {"port": 50053, "name": "Gym Service"},
    "modern_rl": {"port": 50054, "name": "Modern RL Service"},
    "gateway": {"port": 8080, "name": "API Gateway"}
}

BASE_URL = "http://localhost"
TIMEOUT = 10

class TestServiceHealth:
    """Test service health and basic connectivity"""
    
    @pytest.mark.parametrize("service_key", SERVICES.keys())
    def test_service_health_endpoint(self, service_key):
        """Test that each service health endpoint responds correctly"""
        service = SERVICES[service_key]
        url = f"{BASE_URL}:{service['port']}/health"
        
        response = requests.get(url, timeout=TIMEOUT)
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "healthy"
        print(f" {service['name']} is healthy")

class TestGymService:
    """Test Gym service Box2D functionality"""
    
    def test_box2d_availability(self):
        """Test Box2D physics environments availability"""
        url = f"{BASE_URL}:50053/"
        
        response = requests.get(url, timeout=TIMEOUT)
        data = response.json()
        
        if data.get("box2d_available"):
            print(" Box2D physics environments are available!")
        else:
            print(" Box2D physics environments are NOT available")

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "--color=yes"])