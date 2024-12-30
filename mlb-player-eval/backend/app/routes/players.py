from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from ..database import get_db
from ..models.player import Player

router = APIRouter()

@router.get("/players/")
def get_players(year: float = 2024, db: Session = Depends(get_db)):
    try:
        # Log before query
        print(f"\nAttempting to fetch players for year {year}")
        
        # Execute query
        players = db.query(Player).filter(Player.year == year).all()
        
        # Log results
        print(f"Database query completed. Found {len(players)} players")
        if players:
            print(f"First player: {players[0].name}, {players[0].team}")
        else:
            print("No players found in database")
        
        return players
    except Exception as e:
        print(f"Error in get_players: {str(e)}")
        raise
@router.get("/players/{player_name}")
def get_player(player_name: str, db: Session = Depends(get_db)):
    player = db.query(Player).filter(Player.name == player_name).first()
    if player is None:
        raise HTTPException(status_code=404, detail="Player not found")
    return player