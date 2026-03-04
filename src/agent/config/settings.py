import os
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional


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


def _env_bool(key: str, default: bool) -> bool:
    value = os.getenv(key)
    if value is None:
        return default

    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False

    raise ValueError(f"Invalid boolean for {key}: '{value}'")


def _env_list(key: str, default: List[str]) -> List[str]:
    value = os.getenv(key)
    if not value:
        return default
    return [v.strip() for v in value.split(",") if v.strip()]


def _env_float_map(key: str, default: Dict[str, float]) -> Dict[str, float]:
    value = os.getenv(key)
    if not value:
        return dict(default)

    out: Dict[str, float] = {}

    for chunk in value.split(","):
        item = chunk.strip()
        if not item:
            continue

        if ":" not in item:
            raise ValueError(f"Invalid '{key}' entry '{item}'. Expected tool:seconds")

        tool_name, timeout_s = item.split(":", 1)
        name = tool_name.strip()

        if not name:
            raise ValueError(f"Invalid '{key}' entry '{item}': missing tool name")

        out[name] = float(timeout_s.strip())

    return out


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
class ToolPolicySettings:
    allow_tools_by_default: bool = True
    allow_network: bool = True
    allow_filesystem_read: bool = True
    allow_filesystem_write: bool = False

    deny_tools: List[str] = field(default_factory=list)
    allow_tags: List[str] = field(default_factory=list)
    deny_tags: List[str] = field(default_factory=list)
    deny_scopes: List[str] = field(default_factory=list)
    deny_risk: List[str] = field(default_factory=list)

    tool_timeouts_s: Dict[str, float] = field(default_factory=dict)
    enforce_confirmation: bool = False


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
    tool_policy: ToolPolicySettings = field(default_factory=ToolPolicySettings)


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
    emb_model = _env("EMBED_MODEL", "mxbai-x-large")
    emb_dims = _env_int("EMBED_DIMS", 1024)

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

    # ---- Tool policy ----
    tool_policy = ToolPolicySettings(
        allow_tools_by_default=_env_bool("ALLOW_TOOLS_BY_DEFAULT", True),
        allow_network=_env_bool("ALLOW_NETWORK", True),
        allow_filesystem_read=_env_bool("ALLOW_FILESYSTEM_READ", True),
        allow_filesystem_write=_env_bool("ALLOW_FILESYSTEM_WRITE", False),
        deny_tools=_env_list("DENY_TOOLS", default=[]),
        allow_tags=_env_list("ALLOW_TAGS", default=[]),
        deny_tags=_env_list("DENY_TAGS", default=[]),
        deny_scopes=_env_list("DENY_SCOPES", default=[]),
        deny_risk=_env_list("DENY_RISK", default=[]),
        tool_timeouts_s=_env_float_map("TOOL_TIMEOUTS", default={}),
        enforce_confirmation=_env_bool("ENFORCE_TOOL_CONFIRMATION", False),
    )

    settings = Settings(
        env=env,
        debug=debug,
        db=db,
        llm=llm,
        embeddings=embeddings,
        memory=memory,
        workspace=workspace,
        skills_enabled=skills_enabled,
        tool_policy=tool_policy,
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

    allowed_risks = {"safe", "sensitive", "dangerous"}
    invalid_risks = [r for r in settings.tool_policy.deny_risk if r not in allowed_risks]
    if invalid_risks:
        raise ValueError(
            f"Invalid DENY_RISK values: {invalid_risks}. Allowed: safe,sensitive,dangerous"
        )

    for tool_name, timeout_s in settings.tool_policy.tool_timeouts_s.items():
        if timeout_s <= 0:
            raise ValueError(f"Invalid timeout for tool '{tool_name}': must be > 0")
