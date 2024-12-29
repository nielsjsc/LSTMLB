from sqlalchemy import Column, Integer, String, Float
from ..database import Base

class Player(Base):
    __tablename__ = "players"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    team = Column(String)
    status = Column(String)
    year = Column(Float)
    war = Column(Float)
    base_value = Column(Float)
    contract_value = Column(Float)
    surplus_value = Column(Float)

    def __repr__(self):
        return f"<Player {self.name}>"