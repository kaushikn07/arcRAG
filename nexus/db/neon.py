"""
NEXUS Database Connection Module for Neon (Postgres with pgvector)

Provides async connection pool and helper functions for database operations.
"""

import asyncio
from typing import Any, Optional, List, Union
from contextlib import asynccontextmanager

import asyncpg
from asyncpg import Record, Pool
from loguru import logger

from config import settings


class NeonDatabase:
    """Async database connection manager for Neon Postgres."""

    def __init__(self):
        self._pool: Optional[Pool] = None
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the connection pool."""
        if self._initialized:
            return

        try:
            self._pool = await asyncpg.create_pool(
                dsn=settings.neon_database_url,
                max_size=10,
                min_size=2,
                command_timeout=60,
                statement_cache_size=100,
            )
            self._initialized = True
            logger.info("Neon database pool initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize database pool: {e}")
            raise

    async def close(self) -> None:
        """Close the connection pool."""
        if self._pool:
            await self._pool.close()
            self._initialized = False
            logger.info("Neon database pool closed")

    @asynccontextmanager
    async def get_connection(self):
        """Get a connection from the pool."""
        if not self._initialized:
            await self.initialize()

        async with self._pool.acquire() as conn:
            yield conn

    async def fetch(self, query: str, *args: Any) -> List[Record]:
        """
        Execute a SELECT query and return all rows.

        Args:
            query: SQL query string
            *args: Query parameters

        Returns:
            List of Record objects
        """
        if not self._initialized:
            await self.initialize()

        try:
            async with self._pool.acquire() as conn:
                results = await conn.fetch(query, *args)
                return list(results)
        except asyncpg.PostgresError as e:
            logger.error(f"Database fetch error: {e}")
            raise

    async def fetchrow(self, query: str, *args: Any) -> Optional[Record]:
        """
        Execute a query and return the first row or None.

        Args:
            query: SQL query string
            *args: Query parameters

        Returns:
            Single Record object or None
        """
        if not self._initialized:
            await self.initialize()

        try:
            async with self._pool.acquire() as conn:
                result = await conn.fetchrow(query, *args)
                return result
        except asyncpg.PostgresError as e:
            logger.error(f"Database fetchrow error: {e}")
            raise

    async def fetchval(self, query: str, *args: Any, column: int = 0) -> Optional[Any]:
        """
        Execute a query and return the first value of the first row.

        Args:
            query: SQL query string
            *args: Query parameters
            column: Column index to return

        Returns:
            Single value or None
        """
        if not self._initialized:
            await self.initialize()

        try:
            async with self._pool.acquire() as conn:
                result = await conn.fetchval(query, *args, column=column)
                return result
        except asyncpg.PostgresError as e:
            logger.error(f"Database fetchval error: {e}")
            raise

    async def execute(self, query: str, *args: Any) -> str:
        """
        Execute a command (INSERT, UPDATE, DELETE, etc.) and return status.

        Args:
            query: SQL query string
            *args: Query parameters

        Returns:
            Command status string (e.g., 'SELECT 1', 'UPDATE 5')
        """
        if not self._initialized:
            await self.initialize()

        try:
            async with self._pool.acquire() as conn:
                result = await conn.execute(query, *args)
                return result
        except asyncpg.PostgresError as e:
            logger.error(f"Database execute error: {e}")
            raise

    async def executemany(self, query: str, args: List[tuple]) -> None:
        """
        Execute a command with multiple parameter sets.

        Args:
            query: SQL query string
            args: List of parameter tuples
        """
        if not self._initialized:
            await self.initialize()

        try:
            async with self._pool.acquire() as conn:
                await conn.executemany(query, args)
        except asyncpg.PostgresError as e:
            logger.error(f"Database executemany error: {e}")
            raise

    async def transaction(self):
        """Return a transaction context manager."""
        if not self._initialized:
            await self.initialize()
        return self._pool.acquire()


# Global database instance
_db: Optional[NeonDatabase] = None


def get_db() -> NeonDatabase:
    """Get the global database instance."""
    global _db
    if _db is None:
        _db = NeonDatabase()
    return _db


# Convenience functions using the global instance
async def fetch(query: str, *args: Any) -> List[Record]:
    """Fetch all rows from a query."""
    return await get_db().fetch(query, *args)


async def fetchrow(query: str, *args: Any) -> Optional[Record]:
    """Fetch a single row from a query."""
    return await get_db().fetchrow(query, *args)


async def fetchval(query: str, *args: Any, column: int = 0) -> Optional[Any]:
    """Fetch a single value from a query."""
    return await get_db().fetchval(query, *args, column=column)


async def execute(query: str, *args: Any) -> str:
    """Execute a database command."""
    return await get_db().execute(query, *args)


async def executemany(query: str, args: List[tuple]) -> None:
    """Execute a command with multiple parameter sets."""
    return await get_db().executemany(query, args)


async def init_db() -> None:
    """Initialize the global database connection."""
    await get_db().initialize()


async def close_db() -> None:
    """Close the global database connection."""
    await get_db().close()
