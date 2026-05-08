from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "Market Signal Engine"
    debug: bool = True
    read_only_signal_only: bool = True

    # Database
    database_url: str = "sqlite:///market_signals.db"
    db_password: str = ""

    # API Keys
    yfinance_enabled: bool = True
    finnhub_key: str = ""
    binance_enabled: bool = True
    blockchain_enabled: bool = True
    whale_alert_key: str = ""
    dexscreener_enabled: bool = True
    defillama_enabled: bool = True

    # Telegram
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""
    telegram_alerts_enabled: bool = False

    # Scheduling (seconds)
    collector_interval: int = 300
    analysis_interval: int = 600
    signal_check_interval: int = 60

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "allow"}


settings = Settings()
