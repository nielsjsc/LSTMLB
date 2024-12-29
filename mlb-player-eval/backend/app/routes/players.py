from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
from ..database import get_db
from ..models.player import Player

router = APIRouter()

@router.get("/players/")
def get_players(year: float = 2024, db: Session = Depends(get_db)):
    players = db.query(Player).filter(Player.year == year).all()
    return players

@router.get("/players/{player_name}")
def get_player(player_name: str, db: Session = Depends(get_db)):
    player = db.query(Player).filter(Player.name == player_name).first()
    if player is None:
        raise HTTPException(status_code=404, detail="Player not found")
    return player