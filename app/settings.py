from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    RUNS_DIR: str = "runs"
    DEFAULT_MODEL: str = "gpt-4o-mini"
    # LLM cost controls
    LLM_MAX_INPUT_CHARS: int = 9000
    LLM_MAX_OUTPUT_TOKENS: int = 450

settings = Settings()
