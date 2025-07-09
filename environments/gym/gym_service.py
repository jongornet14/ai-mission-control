from fastapi import FastAPI
import uvicorn
import gym
import numpy as np

app = FastAPI(title="AI Mission Control - Gym Service")

@app.get("/health")
def health():
    return {"status": "healthy", "service": "gym"}

@app.get("/")
def root():
    # Test if Box2D environments are available
    box2d_available = True
    try:
        env = gym.make('LunarLander-v2')
        env.close()
    except:
        box2d_available = False
    
    environments = ["CartPole-v1", "MountainCar-v0", "Acrobot-v1"]
    if box2d_available:
        environments.extend(["LunarLander-v2", "BipedalWalker-v3"])
    
    return {
        "message": "AI Mission Control Gym Service",
        "version": "1.0.0",
        "box2d_available": box2d_available,
        "environments": environments
    }

@app.post("/create/{env_id}")
def create_environment(env_id: str):
    try:
        env = gym.make(env_id)
        env.close()
        return {
            "session_id": f"gym_{env_id}_session",
            "env_id": env_id,
            "status": "created"
        }
    except Exception as e:
        return {
            "error": f"Failed to create environment {env_id}: {str(e)}",
            "status": "failed"
        }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=50053)
