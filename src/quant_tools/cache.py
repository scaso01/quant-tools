"""SQLite price cache with atomic writes and thread safety."""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
from datetime import datetime, timedelta

from sqlalchemy import Column, DateTime, Float, Integer, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from quant_tools.config import cfg

log = logging.getLogger("quant_tools")

_lock = threading.Lock()


class Base(DeclarativeBase):
    pass


class PriceCache(Base):
    """Cached price data."""
    __tablename__ = "price_cache"

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False, index=True)
    date = Column(String(10), nullable=False)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    volume = Column(Integer)
    adj_close = Column(Float)
    cached_at = Column(DateTime, default=datetime.now)


class MetadataCache(Base):
    """Cached ticker metadata."""
    __tablename__ = "metadata_cache"

    symbol = Column(String(20), primary_key=True)
    data_json = Column(Text, nullable=False)
    cached_at = Column(DateTime, default=datetime.now)


class PredictionRow(Base):
    """Stored prediction from ML models."""
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, nullable=False, default=datetime.now)
    symbol = Column(String(30), nullable=False, index=True)
    asset_class = Column(String(20), nullable=False)
    model_name = Column(String(50), nullable=False, index=True)
    prediction_direction = Column(String(10), nullable=False)
    predicted_magnitude = Column(Float)
    confidence = Column(Float)
    horizon_days = Column(Integer, nullable=False)
    target_date = Column(String(19), nullable=False)
    interval = Column(String(5), default="1d")
    horizon_hours = Column(Integer, nullable=True)
    consensus_signal = Column(String(20))
    is_ensemble = Column(Integer, default=0)
    resolved = Column(Integer, default=0, index=True)


class PredictionOutcomeRow(Base):
    """Resolved outcome for a prediction."""
    __tablename__ = "prediction_outcomes"

    id = Column(Integer, primary_key=True, autoincrement=True)
    prediction_id = Column(Integer, nullable=False, index=True)
    actual_direction = Column(String(10), nullable=False)
    actual_magnitude = Column(Float)
    correct = Column(Integer, nullable=False)
    error_pct = Column(Float)
    resolved_at = Column(DateTime, nullable=False, default=datetime.now)


class ModelPerformanceRow(Base):
    """Aggregate model performance stats."""
    __tablename__ = "model_performance"

    id = Column(Integer, primary_key=True, autoincrement=True)
    model_name = Column(String(50), nullable=False, index=True)
    asset_class = Column(String(20), nullable=False)
    total_predictions = Column(Integer, default=0)
    wins = Column(Integer, default=0)
    losses = Column(Integer, default=0)
    win_rate = Column(Float, default=0.0)
    avg_confidence = Column(Float, default=0.0)
    avg_error = Column(Float, default=0.0)
    sharpe_of_predictions = Column(Float, default=0.0)
    last_updated = Column(DateTime)


class CacheManager:
    """Thread-safe SQLite cache manager.

    Uses SQLAlchemy for ORM and atomic writes.
    """

    def __init__(self, db_path: str | None = None) -> None:
        self._db_path = db_path or cfg.cache_db_path
        self._engine = create_engine(f"sqlite:///{self._db_path}", echo=False)
        Base.metadata.create_all(self._engine)
        self._migrate(self._engine)
        self._SessionFactory = sessionmaker(bind=self._engine)

    @staticmethod
    def _migrate(engine: object) -> None:
        """Add columns introduced after initial schema."""
        from sqlalchemy import inspect, text
        insp = inspect(engine)
        if "predictions" in insp.get_table_names():
            cols = {c["name"] for c in insp.get_columns("predictions")}
            with engine.begin() as conn:
                if "interval" not in cols:
                    conn.execute(text(
                        "ALTER TABLE predictions ADD COLUMN interval VARCHAR(5) DEFAULT '1d'"
                    ))
                if "horizon_hours" not in cols:
                    conn.execute(text(
                        "ALTER TABLE predictions ADD COLUMN horizon_hours INTEGER"
                    ))

    def _session(self) -> Session:
        return self._SessionFactory()

    def get_prices(
        self, symbol: str, start_date: str, end_date: str
    ) -> list[dict[str, object]] | None:
        """Retrieve cached price data for a symbol and date range.

        Args:
            symbol: Ticker symbol.
            start_date: Start date (YYYY-MM-DD).
            end_date: End date (YYYY-MM-DD).

        Returns:
            List of price dicts or None if not fully cached.
        """
        ttl_cutoff = datetime.now() - timedelta(hours=cfg.cache_ttl_hours)
        with _lock:
            session = self._session()
            try:
                rows = (
                    session.query(PriceCache)
                    .filter(
                        PriceCache.symbol == symbol.upper(),
                        PriceCache.date >= start_date,
                        PriceCache.date <= end_date,
                        PriceCache.cached_at >= ttl_cutoff,
                    )
                    .order_by(PriceCache.date)
                    .all()
                )
                if not rows:
                    return None
                return [
                    {
                        "date": r.date,
                        "open": r.open,
                        "high": r.high,
                        "low": r.low,
                        "close": r.close,
                        "volume": r.volume,
                        "adj_close": r.adj_close,
                    }
                    for r in rows
                ]
            finally:
                session.close()

    def store_prices(self, symbol: str, prices: list[dict[str, object]]) -> None:
        """Store price data in cache. Atomic via SQLAlchemy transaction.

        Args:
            symbol: Ticker symbol.
            prices: List of price bar dicts with date, open, high, low, close, volume.
        """
        sym = symbol.upper()
        with _lock:
            session = self._session()
            try:
                # Remove old entries for this symbol
                session.query(PriceCache).filter(PriceCache.symbol == sym).delete()
                now = datetime.now()
                for p in prices:
                    session.add(PriceCache(
                        symbol=sym,
                        date=str(p["date"]),
                        open=p.get("open"),
                        high=p.get("high"),
                        low=p.get("low"),
                        close=p.get("close"),
                        volume=p.get("volume"),
                        adj_close=p.get("adj_close"),
                        cached_at=now,
                    ))
                session.commit()
            except Exception:
                session.rollback()
                raise
            finally:
                session.close()

    def get_metadata(self, symbol: str) -> dict[str, object] | None:
        """Retrieve cached metadata for a symbol.

        Args:
            symbol: Ticker symbol.

        Returns:
            Metadata dict or None if not cached/expired.
        """
        ttl_cutoff = datetime.now() - timedelta(hours=cfg.cache_ttl_hours)
        with _lock:
            session = self._session()
            try:
                row = (
                    session.query(MetadataCache)
                    .filter(
                        MetadataCache.symbol == symbol.upper(),
                        MetadataCache.cached_at >= ttl_cutoff,
                    )
                    .first()
                )
                if row is None:
                    return None
                return json.loads(row.data_json)
            finally:
                session.close()

    def store_metadata(self, symbol: str, data: dict[str, object]) -> None:
        """Store metadata in cache.

        Args:
            symbol: Ticker symbol.
            data: Metadata dict to cache.
        """
        sym = symbol.upper()
        with _lock:
            session = self._session()
            try:
                existing = session.query(MetadataCache).filter(MetadataCache.symbol == sym).first()
                if existing:
                    existing.data_json = json.dumps(data, default=str)
                    existing.cached_at = datetime.now()
                else:
                    session.add(MetadataCache(
                        symbol=sym,
                        data_json=json.dumps(data, default=str),
                        cached_at=datetime.now(),
                    ))
                session.commit()
            except Exception:
                session.rollback()
                raise
            finally:
                session.close()

    def clear(self) -> None:
        """Clear all cached data."""
        with _lock:
            session = self._session()
            try:
                session.query(PriceCache).delete()
                session.query(MetadataCache).delete()
                session.commit()
            except Exception:
                session.rollback()
                raise
            finally:
                session.close()


def atomic_write_json(filepath: str, data: object) -> None:
    """Write JSON atomically via temp file + os.replace().

    Args:
        filepath: Destination file path.
        data: JSON-serializable data.
    """
    dirpath = os.path.dirname(filepath) or "."
    fd, tmp_path = tempfile.mkstemp(dir=dirpath, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, default=str, separators=(",", ":"))
        os.replace(tmp_path, filepath)
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise
