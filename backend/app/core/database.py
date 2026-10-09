"""
FiniRAG Database Module.

Async SQLAlchemy engine and session management for PostgreSQL.
Every processing attempt creates a new row — no overwrites.
"""

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase

from backend.app.core.config import settings


# Async engine — connection pool managed by SQLAlchemy.
#
# pool_pre_ping drops dead connections instead of failing a request;
# pool_recycle empties connections before a managed Postgres idle cutoff;
# the asyncpg statement_timeout bounds any single query so a slow one cannot
# pin a pooled connection indefinitely (ignored by non-Postgres dialects).
_connect_args: dict = {}
if settings.DATABASE_URL.startswith("postgresql"):
    _connect_args = {"server_settings": {
        "statement_timeout": str(settings.DB_STATEMENT_TIMEOUT_MS)}}

engine = create_async_engine(
    settings.DATABASE_URL,
    echo=False,  # Never echo SQL in production (data protection)
    pool_size=5,
    max_overflow=10,
    pool_pre_ping=True,
    pool_recycle=1800,
    pool_timeout=30,
    connect_args=_connect_args,
)

# Session factory
async_session_factory = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


class Base(DeclarativeBase):
    """Base class for all ORM models."""
    pass


async def get_db() -> AsyncSession:
    """FastAPI dependency — yields an async database session."""
    async with async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def init_db():
    """Create all tables on startup. Idempotent."""
    async with engine.begin() as conn:
        # gen_random_uuid() server default needs pgcrypto on fresh DBs.
        try:
            from sqlalchemy import text as _text

            await conn.execute(_text('CREATE EXTENSION IF NOT EXISTS "pgcrypto"'))
        except Exception:
            pass  # non-Postgres backends (tests) have no extensions
        await conn.run_sync(Base.metadata.create_all)


async def close_db():
    """Dispose engine on shutdown."""
    await engine.dispose()
