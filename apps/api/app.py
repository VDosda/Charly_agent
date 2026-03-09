from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from apps.api.routes.chat import router as chat_router
from apps.api.routes.health import router as health_router


def create_api_app() -> FastAPI:
    app = FastAPI(title="Agent API", version="0.1.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:5173",
            "http://127.0.0.1:5173",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health_router)
    app.include_router(chat_router)
    return app

