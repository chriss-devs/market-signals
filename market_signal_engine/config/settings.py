from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "Market Signal Engine"
    debug: bool = True
    read_only_signal_only: bool = True

    # Database
    database_url: str = "sqlite:///market_signals.db"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
