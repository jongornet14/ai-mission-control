from fastapi import FastAPI
import httpx
import asyncio

app = FastAPI(title="AI Mission Control API Gateway")

# Service endpoints
SERVICES = {
    "trading": "http://trading-service:50051",
    "unity": "http://unity-service:50052", 
    "gym": "http://gym-service:50053",
    "modern_rl": "http://modern-rl-service:50054"
}

@app.get("/health")
def health():
    return {"status": "healthy", "service": "api-gateway"}

@app.get("/")
def root():
    return {
        "message": "AI Mission Control API Gateway", 
        "version": "1.0.0",
        "services": list(SERVICES.keys())
    }

@app.get("/services")
async def list_services():
    """List all available services and their health status"""
    results = {}
    async with httpx.AsyncClient() as client:
        for name, url in SERVICES.items():
            try:
                response = await client.get(f"{url}/health", timeout=5.0)
                results[name] = {
                    "status": "healthy" if response.status_code == 200 else "unhealthy",
                    "url": url
                }
            except:
                results[name] = {"status": "unreachable", "url": url}
    return results

@app.get("/route/{service_name}")
async def route_to_service(service_name: str):
    """Route requests to specific services"""
    if service_name not in SERVICES:
        return {"error": f"Service {service_name} not found"}
    
    service_url = SERVICES[service_name]
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(f"{service_url}/health")
            return {"service": service_name, "response": response.json()}
        except Exception as e:
            return {"error": f"Failed to contact {service_name}: {str(e)}"}
