from openenv.core.env_server import create_fastapi_app
from models import DBAction, DBObservation
from .environment import DBEnvironment

app = create_fastapi_app(DBEnvironment, DBAction, DBObservation)
import uvicorn

def main():
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()