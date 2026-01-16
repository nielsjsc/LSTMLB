from app.database import SessionLocal
from app.models.player import Player
from sqlalchemy import desc, asc

def debug_exact_api_query():
    db = SessionLocal()
    try:
        print("=== TESTING EXACT API QUERY ===")
        
        # This should match your API query exactly
        query = db.query(Player).filter(Player.year == 2025)
        print(f"Step 1 - After year filter: {query.count()} players")
        
        query = query.filter(Player.trade_value.isnot(None))
        print(f"Step 2 - After NOT NULL filter: {query.count()} players")
        
        query = query.filter(Player.trade_value != float('nan'))
        print(f"Step 3 - After NaN filter: {query.count()} players")
        
        # Apply sorting like your API
        query = query.order_by(desc(Player.trade_value))
        print(f"Step 4 - After sorting: {query.count()} players")
        
        # Apply pagination like your API
        page = 1
        page_size = 50
        offset = (page - 1) * page_size
        players = query.offset(offset).limit(page_size).all()
        print(f"Step 5 - After pagination: {len(players)} players returned")
        
        if len(players) > 0:
            print("\n=== FIRST FEW RESULTS ===")
            for i, player in enumerate(players[:5]):
                print(f"{i+1}. {player.name}: {player.trade_value}")
        else:
            print("\n!!! NO PLAYERS RETURNED !!!")
            
            # Let's see what the NaN filter is doing
            print("\nTesting NaN filter specifically...")
            test_query = db.query(Player).filter(
                Player.year == 2025,
                Player.trade_value.isnot(None)
            )
            before_nan = test_query.count()
            
            after_nan = test_query.filter(Player.trade_value != float('nan')).count()
            print(f"Before NaN filter: {before_nan}")
            print(f"After NaN filter: {after_nan}")
            print(f"NaN filter removed: {before_nan - after_nan} players")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        db.close()

if __name__ == "__main__":
    debug_exact_api_query()