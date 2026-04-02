from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from typing import Optional
import json
import threading
from .environment import DBEnvironment

app = FastAPI(title="OpenEnv Simulation Server")

# Thread lock for REST endpoints (WebSocket sessions have their own instances)
_lock = threading.Lock()


def _obs_to_payload(obs) -> dict:
    """Convert a DBObservation (Pydantic model) to the wire format EnvClient expects."""
    d = obs.model_dump() if hasattr(obs, "model_dump") else vars(obs)
    return {
        "observation": {
            "current_indices": d.get("current_indices", []),
            "query_cost":      d.get("query_cost",      0.0),
            "storage_used":    d.get("storage_used",    0.0),
            "storage_budget":  d.get("storage_budget",  0.0),
            "message":         d.get("message",         ""),
        },
        "done":   d.get("done",   False),
        "reward": d.get("reward", 0.0),
    }


# ─────────────────────────────────────────────────────────────
# REST Endpoints (used by hackathon checker & /query)
# ─────────────────────────────────────────────────────────────

@app.get("/")
def read_root():
    return {"status": "running", "environment": "OpenEnv Simulation"}

@app.get("/query")
def get_active_query():
    tmp = DBEnvironment()
    return {"query": tmp.get_active_query()}


class _DBActionBody(BaseModel):
    command:    str
    table_name: str
    column_name: str


# ─────────────────────────────────────────────────────────────
# WebSocket Endpoint  /ws
# Compatible with openenv-core EnvClient
#
# Protocol (client → server):
#   {"type": "reset", "data": {"task": "easy"}}
#   {"type": "step",  "data": {"command": ..., "table_name": ..., "column_name": ...}}
#   {"type": "state"}
#   {"type": "close"}
#
# Protocol (server → client):
#   {"data": {"observation": {...}, "done": bool, "reward": float}}
#   {"data": {"episode_id": ..., "step_count": ..., "max_steps": ...}}   ← for state
#   {"type": "error", "data": {"message": ..., "code": ...}}
# ─────────────────────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    # Each WebSocket session owns its own isolated environment
    from models import DBAction as ModelAction
    env = DBEnvironment()

    try:
        while True:
            raw  = await websocket.receive_text()
            msg  = json.loads(raw)
            mtype = msg.get("type", "")
            data  = msg.get("data", {})

            if mtype == "reset":
                task = data.get("task", data.get("task_name", "easy"))
                obs  = env.reset(task=task)
                await websocket.send_json({"data": _obs_to_payload(obs)})

            elif mtype == "step":
                action = ModelAction(
                    command=     data.get("command",     "FINISH"),
                    table_name=  data.get("table_name",  "users"),
                    column_name= data.get("column_name", ""),
                )
                obs = env.step(action)
                await websocket.send_json({"data": _obs_to_payload(obs)})

            elif mtype == "state":
                s = env.state
                await websocket.send_json({
                    "data": {
                        "episode_id": s.episode_id,
                        "step_count": s.step_count,
                        "max_steps":  s.max_steps,
                    }
                })

            elif mtype == "close":
                await websocket.close()
                break

            else:
                await websocket.send_json({
                    "type": "error",
                    "data": {"message": f"Unknown type: {mtype}", "code": "UNKNOWN_TYPE"},
                })

    except WebSocketDisconnect:
        pass  # client left cleanly
    except Exception as exc:
        try:
            await websocket.send_json({
                "type": "error",
                "data": {"message": str(exc), "code": "SERVER_ERROR"},
            })
        except Exception:
            pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)