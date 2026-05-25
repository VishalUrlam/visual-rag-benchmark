from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    supermemory_api_key: str = ""
    supermemory_base_url: str = "https://api.supermemory.ai/v3"
    anthropic_api_key: str = ""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


settings = Settings()
