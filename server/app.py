from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
import os
import json
import threading
from environment import DatabaseEnvironment

app = FastAPI(title="OpenEnv Simulation Server")

# Global environment and thread-safety lock
env = DatabaseEnvironment()
lock = threading.Lock()

class DBAction(BaseModel):
    thought_process: str
    command: str
    table_name: str
    column_name: str

@app.get("/")
def read_root():
    return {"status": "running", "environment": "OpenEnv Simulation"}

@app.get("/state")
def get_state():
    return env.get_state()

@app.post("/reset")
def reset_env(task_name: str = "easy"):
    with lock:
        return env.reset(task_name)

@app.post("/action")
def take_action(action: DBAction):
    with lock:
        # Convert Pydantic model to dict for environment processing
        action_dict = action.model_dump() if hasattr(action, 'model_dump') else action.dict()
        return env.step(action_dict)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)