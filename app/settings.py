import os
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    APP_NAME: str = Field(default=os.getenv("APP_NAME", "voice-assessment"))
    ENVIRONMENT: str = Field(default=os.getenv("ENVIRONMENT", "local"))
    GOOGLE_API_KEY: str = Field(default=os.getenv("GOOGLE_API_KEY", ""))
    HF_TOKEN: str = Field(default=os.getenv("HF_TOKEN", ""))
    WHISPER_MODEL: str = Field(default=os.getenv("WHISPER_MODEL", "base"))
    WHISPER_DEVICE: str = Field(default=os.getenv("WHISPER_DEVICE", "cpu"))
    WHISPER_COMPUTE_TYPE: str = Field(default=os.getenv("WHISPER_COMPUTE_TYPE", "int8"))
    ALLOW_ORIGINS: str = Field(default=os.getenv("ALLOW_ORIGINS", "*"))
    TEMP_DIR: str = Field(default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "temp_audio"))
    CACHE_DIR: str = Field(default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "result_cache"))
    CACHE_TTL_DAYS: int = Field(default=7)

    model_config = {
        "env_file": (
            os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), ".env"),
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"),
            ".env"
        ),
        "extra": "ignore"
    }


settings = Settings()
