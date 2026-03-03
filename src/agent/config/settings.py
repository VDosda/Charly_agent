import os
from dataclasses import dataclass, field
from typing import List, Literal, Optional


LLMProviderType = Literal["openai", "anthropic", "ollama"]
EmbeddingProviderType = Literal["openai", "ollama", "local"]
VecExtensionType = Literal["sqlite_vec", "sqlite_vss", "none"]


def _env(key: str, default: Optional[str] = None) -> Optional[str]:
    return os.getenv(key, default)


def _env_int(key: str, default: int) -> int:
    value = os.getenv(key)
    return int(value) if value is not None else default


def _env_float(key: str, default: float) -> float:
    value = os.getenv(key)
    return float(value) if value is not None else default


def _env_list(key: str, default: List[str]) -> List[str]:
    value = os.getenv(key)
    if not value:
        return default
    return [v.strip() for v in value.split(",") if v.strip()]


# =========================
# Dataclasses
# =========================

@dataclass
class LLMSettings:
    provider: LLMProviderType
    model: str
    temperature: float = 0.2
    max_tokens: int = 2048
    api_key: Optional[str] = None
    base_url: Optional[str] = None  # useful for Ollama or proxy


@dataclass
class EmbeddingSettings:
    provider: EmbeddingProviderType
    model: str
    dimensions: int


@dataclass
class MemorySettings:
    # ST
    st_active_turns: int = 30

    # MT
    mt_turn_window: int = 12
    mt_ttl_days: int = 7
    mt_max_active_episodes: int = 5

    # LT
    lt_confidence_threshold: float = 0.7
    lt_importance_threshold: float = 0.8


@dataclass
class DatabaseSettings:
    path: str
    vec_extension: VecExtensionType
    wal_mode: bool = True
    foreign_keys: bool = True


@dataclass
class Settings:
    env: str
    debug: bool

    db: DatabaseSettings
    llm: LLMSettings
    embeddings: EmbeddingSettings
    memory: MemorySettings

    workspace: str

    skills_enabled: List[str] = field(default_factory=list)


# =========================
# Loader
# =========================

def load_settings() -> Settings:
    env = _env("APP_ENV", "dev")
    debug = _env("APP_DEBUG", "true").lower() == "true"

    # ---- Database ----
    db_path = _env("DB_PATH", "agent.db")
    vec_extension = _env("VEC_EXTENSION", "sqlite_vec")

    db = DatabaseSettings(
        path=db_path,
        vec_extension=vec_extension,
    )

    # ---- LLM ----
    llm_provider: LLMProviderType = _env("LLM_PROVIDER", "ollama")  # default local
    llm_model = _env("LLM_MODEL", "llama3")
    llm_temperature = _env_float("LLM_TEMPERATURE", 0.2)
    llm_max_tokens = _env_int("LLM_MAX_TOKENS", 2048)

    llm = LLMSettings(
        provider=llm_provider,
        model=llm_model,
        temperature=llm_temperature,
        max_tokens=llm_max_tokens,
        api_key=_env("LLM_API_KEY"),
        base_url=_env("LLM_BASE_URL"),
    )

    # ---- Embeddings ----
    emb_provider: EmbeddingProviderType = _env("EMBED_PROVIDER", "ollama")
    emb_model = _env("EMBED_MODEL", "nomic-embed-text")
    emb_dims = _env_int("EMBED_DIMS", 768)

    embeddings = EmbeddingSettings(
        provider=emb_provider,
        model=emb_model,
        dimensions=emb_dims,
    )

    # ---- Memory ----
    memory = MemorySettings(
        st_active_turns=_env_int("ST_ACTIVE_TURNS", 30),
        mt_turn_window=_env_int("MT_TURN_WINDOW", 12),
        mt_ttl_days=_env_int("MT_TTL_DAYS", 7),
        mt_max_active_episodes=_env_int("MT_MAX_EPISODES", 5),
        lt_confidence_threshold=_env_float("LT_CONF_THRESHOLD", 0.7),
        lt_importance_threshold=_env_float("LT_IMPORT_THRESHOLD", 0.8),
    )

    # ---- Skills ----
    skills_enabled = _env_list(
        "SKILLS_ENABLED",
        default=[
            "builtins.time",
            "builtins.http",
            "builtins.filesystem",
        ],
    )
    workspace = _env("AGENT_WORKSPACE", "workspace")

    settings = Settings(
        env=env,
        debug=debug,
        db=db,
        llm=llm,
        embeddings=embeddings,
        memory=memory,
        workspace=workspace,
        skills_enabled=skills_enabled,
    )

    _validate_settings(settings)

    return settings


# =========================
# Validate Settings
# =========================

def _validate_settings(settings: Settings):
    if settings.embeddings.dimensions <= 0:
        raise ValueError("Embedding dimensions must be > 0")

    if settings.memory.mt_turn_window <= 0:
        raise ValueError("MT turn window must be > 0")

    if settings.memory.lt_confidence_threshold > 1:
        raise ValueError("Confidence threshold must be <= 1")

    if settings.memory.lt_importance_threshold > 1:
        raise ValueError("Importance threshold must be <= 1")

    if settings.db.vec_extension not in {"sqlite_vec", "sqlite_vss", "none"}:
        raise ValueError("Invalid vector extension type")
