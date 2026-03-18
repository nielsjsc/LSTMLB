"""DB model for trade-value history (year-by-year surplus timeline per player)."""

from sqlalchemy import Column, Integer, Float, String, Index
from app.database import Base


class TradeValueHistory(Base):
    __tablename__ = "trade_value_history"

    id = Column(Integer, primary_key=True, autoincrement=True)
    mlb_id = Column(Integer, nullable=False, index=True)
    idfg = Column(Integer, nullable=True)
    name = Column(String, nullable=True)
    year = Column(Integer, nullable=False)
    value = Column(Float, nullable=True)
    value_type = Column(String, nullable=True)
    label = Column(String, nullable=True)

    __table_args__ = (
        Index("ix_tvh_mlb_year", "mlb_id", "year"),
    )
