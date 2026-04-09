"""
server/app.py — NOVA OpenEnv Server

FastAPI server with REST + WebSocket endpoints for the OpenEnv protocol.
Each WebSocket session owns an isolated DBEnvironment instance.
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, PlainTextResponse
from pydantic import BaseModel
from typing import Optional
import json
import threading
import os

from .environment import DBEnvironment

app = FastAPI(
    title="NOVA DBA Optimization Environment",
    description="Procedurally-generated database optimization environment for RL agents",
    version="2.0.0",
)

_lock = threading.Lock()


def _obs_to_payload(obs) -> dict:
    """Convert DBObservation to the wire format EnvClient expects."""
    d = obs.model_dump() if hasattr(obs, "model_dump") else vars(obs)
    return {
        "observation": {
            "current_indices": d.get("current_indices", []),
            "query_cost":      d.get("query_cost", 0.0),
            "storage_used":    d.get("storage_used", 0.0),
            "storage_budget":  d.get("storage_budget", 0.0),
            "message":         d.get("message", ""),
            # Rich context fields
            "target_query":    d.get("target_query", ""),
            "table_schemas":   d.get("table_schemas", {}),
            "query_plan":      d.get("query_plan", ""),
            "row_counts":      d.get("row_counts", {}),
            "index_details":   d.get("index_details", []),
            "valid_actions":   d.get("valid_actions", []),
            "difficulty":      d.get("difficulty", ""),
            "scenario_id":     d.get("scenario_id", ""),
        },
        "done":   d.get("done", False),
        "reward": d.get("reward", 0.0),
    }


# ─────────────────────────────────────────────────────────────
# REST Endpoints
# ─────────────────────────────────────────────────────────────

@app.get("/raw_readme")
def get_readme():
    try:
        with open("README.md", "r") as f:
            return PlainTextResponse(f.read())
    except Exception:
        return PlainTextResponse("# NOVA DBA Optimization Environment")


@app.get("/", response_class=HTMLResponse)
def read_root():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>NOVA DBA Agent — Procedural Optimization Environment</title>
        <script type="module" src="https://cdn.jsdelivr.net/gh/zerodevx/zero-md@2/dist/zero-md.min.js"></script>
        <style>
            body { background-color: #0d1117; padding: 40px 20px; margin: 0; font-family: sans-serif; }
            .container { max-width: 900px; margin: 0 auto; }
        </style>
    </head>
    <body>
        <div class="container">
            <zero-md src="/raw_readme">
                <template>
                    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/github-markdown-css/5.2.0/github-markdown-dark.min.css" />
                    <style> .markdown-body { background: transparent !important; } </style>
                </template>
            </zero-md>
        </div>
    </body>
    </html>
    """


@app.get("/query")
def get_active_query(task: Optional[str] = "easy"):
    """Return the SQL query being optimized in the current scenario."""
    with _lock:
        env = DBEnvironment()
        env.reset(task=task)
        return {"query": env.get_active_query(), "scenario_id": env.scenario_id}


@app.post("/reset")
def reset_environment(data: dict = {}):
    """OpenEnv health-check endpoint."""
    return {"status": "ok", "version": "2.0.0"}


@app.get("/scenario_sample")
def scenario_sample(task: Optional[str] = "easy", count: int = 3):
    """
    Generate sample scenarios to demonstrate procedural diversity.
    This endpoint showcases that each reset creates a unique scenario.
    """
    samples = []
    for i in range(min(count, 10)):
        env = DBEnvironment()
        obs = env.reset(task=task)
        d = obs.model_dump() if hasattr(obs, "model_dump") else vars(obs)
        samples.append({
            "scenario_id": d.get("scenario_id", ""),
            "query": d.get("target_query", ""),
            "tables": list(d.get("table_schemas", {}).keys()),
            "row_counts": d.get("row_counts", {}),
            "existing_indices": d.get("current_indices", []),
            "storage_budget": d.get("storage_budget", 0),
            "initial_cost": d.get("query_cost", 0),
        })
    return {"task": task, "samples": samples}


# ─────────────────────────────────────────────────────────────
# WebSocket Endpoint /ws — OpenEnv Protocol
# ─────────────────────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    from models import DBAction as ModelAction
    env = DBEnvironment()

    try:
        while True:
            raw = await websocket.receive_text()
            msg = json.loads(raw)
            mtype = msg.get("type", "")
            data = msg.get("data", {})

            if mtype == "reset":
                task = data.get("task", data.get("task_name", "easy"))
                seed = data.get("seed", None)
                obs = env.reset(task=task, seed=seed)
                await websocket.send_json({"data": _obs_to_payload(obs)})

            elif mtype == "step":
                action = ModelAction(
                    command=data.get("command", "FINISH"),
                    table_name=data.get("table_name", ""),
                    column_name=data.get("column_name", ""),
                )
                obs = env.step(action)
                await websocket.send_json({"data": _obs_to_payload(obs)})

            elif mtype == "state":
                s = env.state
                await websocket.send_json({
                    "data": {
                        "episode_id": s.episode_id,
                        "step_count": s.step_count,
                        "max_steps": s.max_steps,
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
        pass
    except Exception as exc:
        try:
            await websocket.send_json({
                "type": "error",
                "data": {"message": str(exc), "code": "SERVER_ERROR"},
            })
        except Exception:
            pass


def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()