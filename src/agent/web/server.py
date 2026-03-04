from __future__ import annotations

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv

from agent.config.settings import load_settings
from agent.db.conn import get_connection
from agent.db.migrate import migrate
from agent.db.healthcheck import healthcheck_db
from agent.web.api.traces import router as traces_router

def create_web_app() -> FastAPI:
    # Keep monitor DB boot aligned with the main app:
    # load .env, then ensure schema is migrated before serving endpoints.
    load_dotenv()
    settings = load_settings()

    conn = get_connection(settings.db.path, settings.db.vec_extension)
    try:
        migrate(conn)
        healthcheck_db(conn)
    finally:
        conn.close()

    app = FastAPI(title="Agent Monitor", version="0.1")

    app.include_router(traces_router, prefix="/api")

    # Serve static UI
    app.mount("/", StaticFiles(directory="src/agent/web/ui", html=True), name="ui")
    return app
