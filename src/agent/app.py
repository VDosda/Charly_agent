# src/agent/app.py

from agent.config.settings import load_settings
from agent.db.conn import get_connection
from agent.db.migrate import migrate
from agent.db.healthcheck import healthcheck_db

from agent.providers.llm import get_llm_provider
from agent.providers.embeddings import get_embedding_provider

from agent.skills.registry import load_enabled_skills

from agent.core.runtime import AgentRuntime

from dotenv import load_dotenv

load_dotenv()  # loads .env from current working directory by default


def create_runtime():
    # 1. Load config
    settings = load_settings()

    # 2. DB
    conn = get_connection(settings.db.path, settings.db.vec_extension)
    migrate(conn)
    healthcheck_db(conn)

    # 3. Providers
    llm = get_llm_provider(settings)
    embeddings = get_embedding_provider(settings)

    # 4. Skills
    skills = load_enabled_skills(settings.skills_enabled, settings)

    # 5. Runtime
    runtime = AgentRuntime(
        db=conn,
        llm=llm,
        embeddings=embeddings,
        skills=skills,
        settings=settings,
    )

    return runtime


def create_api_app():
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware

    from agent.api.chat import router as chat_router
    from agent.api.health import router as health_router

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


def main():
    runtime = create_runtime()

    print("Agent ready. Type 'exit' to quit.\n")

    session_id = "cli-session"
    user_id = "local-user"

    while True:
        user_input = input(">> ")

        if user_input.strip().lower() in {"exit", "quit"}:
            break

        response = runtime.handle_message(
            user_id=user_id,
            session_id=session_id,
            message=user_input,
        )

        print(response)


if __name__ == "__main__":
    main()
