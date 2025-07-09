#!/usr/bin/env python3
"""
AI Mission Control - Complete Gym Service
Fully functional gym service with all RL endpoints implemented
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
# import gym
import numpy as np
import uuid
import logging
from typing import Dict, Any, Optional, List
import uvicorn
from datetime import datetime
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="AI Mission Control - Gym Service", version="1.0.0")

# Global session storage
active_sessions: Dict[str, Dict[str, Any]] = {}

# Request/Response models
class EnvironmentConfig(BaseModel):
    config: Dict[str, Any] = {}

class ActionRequest(BaseModel):
    action: Any

class StepResponse(BaseModel):
    observation: List[float]
    reward: float
    done: bool
    info: Dict[str, Any]

class ResetResponse(BaseModel):
    observation: List[float]
    info: Dict[str, Any] = {}

class CreateResponse(BaseModel):
    session_id: str
    env_name: str
    observation_space: Dict[str, Any]
    action_space: Dict[str, Any]
    message: str

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    service: str
    version: str

def check_box2d_availability():
    """Check if Box2D physics environments are available"""
    try:
        import Box2D
        return True
    except ImportError:
        return False

def get_available_environments():
    """Get list of available gym environments"""
    basic_envs = [
        "CartPole-v1",
        "MountainCar-v0", 
        "Acrobot-v1",
        "Pendulum-v1"
    ]
    
    box2d_envs = [
        "LunarLander-v2",
        "BipedalWalker-v3",
        "CarRacing-v2"
    ]
    
    atari_envs = [
        "ALE/Breakout-v5",
        "ALE/Pong-v5",
        "ALE/SpaceInvaders-v5"
    ]
    
    envs = basic_envs.copy()
    
    if check_box2d_availability():
        envs.extend(box2d_envs)
        
    # Check if Atari environments are available
    try:
        import ale_py
        envs.extend(atari_envs)
    except ImportError:
        pass
        
    return envs

def serialize_space(space):
    """Serialize gym spaces for JSON response"""
    if isinstance(space, gym.spaces.Box):
        return {
            "type": "Box",
            "shape": list(space.shape),
            "low": space.low.tolist() if hasattr(space.low, 'tolist') else float(space.low),
            "high": space.high.tolist() if hasattr(space.high, 'tolist') else float(space.high),
            "dtype": str(space.dtype)
        }
    elif isinstance(space, gym.spaces.Discrete):
        return {
            "type": "Discrete", 
            "n": space.n
        }
    elif isinstance(space, gym.spaces.MultiDiscrete):
        return {
            "type": "MultiDiscrete",
            "nvec": space.nvec.tolist()
        }
    else:
        return {
            "type": str(type(space).__name__),
            "details": str(space)
        }

@app.get("/")
async def root():
    """Root endpoint with service information"""
    return {
        "service": "AI Mission Control - Gym Service",
        "version": "1.0.0",
        "status": "healthy",
        "environments": get_available_environments(),
        "box2d_available": check_box2d_availability(),
        "active_sessions": len(active_sessions),
        "capabilities": {
            "create_environment": True,
            "reset_environment": True,
            "step_environment": True,
            "close_environment": True,
            "render": False,  # Rendering disabled in headless mode
            "record": False
        },
        "message": "OpenAI Gym microservice ready for RL training",
        "endpoints": [
            "GET / - Service information",
            "GET /health - Health check",
            "POST /create/{env_name} - Create environment",
            "POST /reset/{session_id} - Reset environment", 
            "POST /step/{session_id} - Step environment",
            "DELETE /close/{session_id} - Close environment",
            "GET /sessions - List active sessions"
        ]
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        service="gym-service",
        version="1.0.0"
    )

@app.post("/create/{env_name}", response_model=CreateResponse)
async def create_environment(env_name: str, config: EnvironmentConfig = EnvironmentConfig()):
    """Create a new gym environment instance"""
    try:
        # Validate environment name
        available_envs = get_available_environments()
        if env_name not in available_envs:
            raise HTTPException(
                status_code=400, 
                detail=f"Environment '{env_name}' not available. Available: {available_envs}"
            )
        
        # Create environment
        logger.info(f"Creating environment: {env_name}")
        env = gym.make(env_name)
        
        # Generate session ID
        session_id = str(uuid.uuid4())
        
        # Store session
        active_sessions[session_id] = {
            "env": env,
            "env_name": env_name,
            "created_at": datetime.now().isoformat(),
            "step_count": 0,
            "episode_count": 0,
            "total_reward": 0.0,
            "config": config.config,
            "last_action": None,
            "done": True  # Start with done=True, requires reset
        }
        
        logger.info(f"Environment {env_name} created with session ID: {session_id}")
        
        return CreateResponse(
            session_id=session_id,
            env_name=env_name,
            observation_space=serialize_space(env.observation_space),
            action_space=serialize_space(env.action_space),
            message=f"Environment {env_name} created successfully"
        )
        
    except Exception as e:
        logger.error(f"Failed to create environment {env_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create environment: {str(e)}")

@app.post("/reset/{session_id}", response_model=ResetResponse)
async def reset_environment(session_id: str):
    """Reset environment to initial state"""
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    
    try:
        session = active_sessions[session_id]
        env = session["env"]
        
        # Reset environment
        observation, info = env.reset()
        
        # Update session state
        session["step_count"] = 0
        session["episode_count"] += 1
        session["done"] = False
        session["last_observation"] = observation.tolist()
        
        logger.info(f"Reset environment {session['env_name']} (session: {session_id[:8]})")
        
        return ResetResponse(
            observation=observation.tolist(),
            info=info or {}
        )
        
    except Exception as e:
        logger.error(f"Failed to reset environment {session_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to reset environment: {str(e)}")

@app.post("/step/{session_id}", response_model=StepResponse)
async def step_environment(session_id: str, action_request: ActionRequest):
    """Execute one step in the environment"""
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    
    session = active_sessions[session_id]
    
    # Check if environment needs reset
    if session.get("done", True):
        raise HTTPException(
            status_code=400, 
            detail="Environment is done. Call reset before stepping."
        )
    
    try:
        env = session["env"]
        action = action_request.action
        
        # Convert action to appropriate type
        if isinstance(env.action_space, gym.spaces.Discrete):
            action = int(action)
        elif isinstance(env.action_space, gym.spaces.Box):
            action = np.array(action, dtype=env.action_space.dtype)
        
        # Execute step
        observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # Update session state
        session["step_count"] += 1
        session["total_reward"] += reward
        session["done"] = done
        session["last_action"] = action
        session["last_observation"] = observation.tolist()
        
        logger.info(f"Step {session['step_count']} in {session['env_name']}: "
                   f"action={action}, reward={reward:.3f}, done={done}")
        
        return StepResponse(
            observation=observation.tolist(),
            reward=float(reward),
            done=done,
            info=info or {}
        )
        
    except Exception as e:
        logger.error(f"Failed to step environment {session_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to step environment: {str(e)}")

@app.delete("/close/{session_id}")
async def close_environment(session_id: str):
    """Close and cleanup environment"""
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    
    try:
        session = active_sessions[session_id]
        env = session["env"]
        env_name = session["env_name"]
        
        # Close environment
        env.close()
        
        # Remove from active sessions
        del active_sessions[session_id]
        
        logger.info(f"Closed environment {env_name} (session: {session_id[:8]})")
        
        return {
            "message": f"Environment {env_name} closed successfully",
            "session_id": session_id
        }
        
    except Exception as e:
        logger.error(f"Failed to close environment {session_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to close environment: {str(e)}")

@app.get("/sessions")
async def list_active_sessions():
    """List all active sessions"""
    sessions_info = []
    
    for session_id, session in active_sessions.items():
        sessions_info.append({
            "session_id": session_id,
            "env_name": session["env_name"],
            "created_at": session["created_at"],
            "step_count": session["step_count"],
            "episode_count": session["episode_count"],
            "total_reward": session["total_reward"],
            "done": session.get("done", True)
        })
    
    return {
        "active_sessions": len(active_sessions),
        "sessions": sessions_info
    }

@app.get("/session/{session_id}")
async def get_session_details(session_id: str):
    """Get detailed information about a specific session"""
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    
    session = active_sessions[session_id]
    env = session["env"]
    
    return {
        "session_id": session_id,
        "env_name": session["env_name"],
        "created_at": session["created_at"],
        "step_count": session["step_count"],
        "episode_count": session["episode_count"],
        "total_reward": session["total_reward"],
        "done": session.get("done", True),
        "last_action": session.get("last_action"),
        "last_observation": session.get("last_observation"),
        "config": session["config"],
        "observation_space": serialize_space(env.observation_space),
        "action_space": serialize_space(env.action_space)
    }

# Cleanup function for graceful shutdown
@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup all environments on shutdown"""
    logger.info("Shutting down gym service, cleaning up environments...")
    
    for session_id, session in active_sessions.items():
        try:
            session["env"].close()
            logger.info(f"Closed environment {session['env_name']} (session: {session_id[:8]})")
        except Exception as e:
            logger.error(f"Error closing environment {session_id}: {e}")
    
    active_sessions.clear()
    logger.info("Gym service shutdown complete")

if __name__ == "__main__":
    # Run the service
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=50053,
        log_level="info"
    )