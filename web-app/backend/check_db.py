from app.database import SessionLocal
from app.models.player import Player

session = SessionLocal()
player = session.query(Player).filter(
    Player.name.ilike('%Yordan Alvarez%'),
    Player.year == 2026,
    Player.projection_type == 'ros'
).first()

if player:
    print('Yordan Alvarez 2026 (DB):')
    print(f'  years_control: {player.years_control} (type: {type(player.years_control).__name__})')
    print(f'  fa_year: {player.fa_year}')
    print(f'  control_through: {player.control_through}')
    print(f'  trade_value: {player.trade_value}')
else:
    print('Player not found')
session.close()
