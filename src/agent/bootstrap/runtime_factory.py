from dotenv import load_dotenv

from agent.application.runtime import AgentRuntime
from agent.bootstrap.settings import load_settings
from agent.infrastructure.db.conn import get_connection
from agent.infrastructure.db.healthcheck import healthcheck_db
from agent.infrastructure.db.migrate import migrate
from agent.infrastructure.providers.embeddings import get_embedding_provider
from agent.infrastructure.providers.llm import get_llm_provider
from agent.infrastructure.skills.registry import load_enabled_skills

load_dotenv()


def create_runtime() -> AgentRuntime:
    settings = load_settings()

    conn = get_connection(settings.db.path, settings.db.vec_extension)
    migrate(conn)
    healthcheck_db(conn)

    llm = get_llm_provider(settings)
    embeddings = get_embedding_provider(settings)
    skills = load_enabled_skills(settings.skills_enabled, settings)

    return AgentRuntime(
        db=conn,
        llm=llm,
        embeddings=embeddings,
        skills=skills,
        settings=settings,
    )

