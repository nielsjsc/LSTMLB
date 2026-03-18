"""DB model for Spotrac contract transactions."""

from sqlalchemy import Column, Integer, Float, String, Index
from app.database import Base


class SpotracTransaction(Base):
    __tablename__ = "spotrac_transactions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    spotrac_id = Column(Integer, nullable=True)
    player_name = Column(String, nullable=False)
    player_name_lower = Column(String, nullable=False, index=True)
    date = Column(String, nullable=True)
    transaction_type = Column(String, nullable=True)
    description = Column(String, nullable=True)
    team = Column(String, nullable=True)
    years = Column(Float, nullable=True)
    total_value = Column(Float, nullable=True)
    annual_value = Column(Float, nullable=True)
