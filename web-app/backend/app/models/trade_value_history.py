"""DB model for trade-value history (date-level surplus timeline per player)."""

from sqlalchemy import Column, Integer, Float, String, Index
from app.database import Base


class TradeValueHistory(Base):
    __tablename__ = "trade_value_history"

    id = Column(Integer, primary_key=True, autoincrement=True)
    mlb_id = Column(Integer, nullable=False, index=True)
    idfg = Column(Integer, nullable=True)
    name = Column(String, nullable=True)
    date = Column(String, nullable=True)           # YYYY-MM-DD
    year = Column(Integer, nullable=False)
    value = Column(Float, nullable=True)
    value_type = Column(String, nullable=True)
    transaction_type = Column(String, nullable=True)  # Spotrac txn type
    label = Column(String, nullable=True)

    __table_args__ = (
        Index("ix_tvh_mlb_date", "mlb_id", "date"),
        Index("ix_tvh_mlb_year", "mlb_id", "year"),
    )
