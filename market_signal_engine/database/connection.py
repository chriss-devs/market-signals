from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from market_signal_engine.config.settings import settings

_engine: Engine | None = None
_session_factory: sessionmaker[Session] | None = None


def get_engine() -> Engine:
    global _engine
    if _engine is None:
        _engine = create_engine(
            settings.database_url,
            echo=settings.debug,
            connect_args={"check_same_thread": False} if "sqlite" in settings.database_url else {},
            pool_pre_ping=True,
        )
    return _engine


def get_session() -> Session:
    global _session_factory
    if _session_factory is None:
        _session_factory = sessionmaker(bind=get_engine())
    return _session_factory()


def init_db() -> None:
    """Create all tables. For dev only; production uses Alembic."""
    from market_signal_engine.database.models import Base
    Base.metadata.create_all(bind=get_engine())
