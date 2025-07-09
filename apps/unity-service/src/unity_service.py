from fastapi import FastAPI
import uvicorn

app = FastAPI(title="AI Mission Control - Unity Service")

@app.get("/health")
def health():
    return {"status": "healthy", "service": "unity"}

@app.get("/")
def root():
    return {
        "message": "AI Mission Control Unity Service",
        "version": "1.0.0",
        "environments": ["unity_game.exe", "custom_unity_env"]
    }

@app.post("/create/{env_id}")
def create_environment(env_id: str):
    return {
        "session_id": f"unity_{env_id}_session",
        "env_id": env_id,
        "status": "created"
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=50052)
