"""DB model for evaluated past trades.

Stores the fully-augmented trade data (projections + historical WAR +
prospect values) so the web server reads directly from DB without any
startup processing.

The *sides_json* column holds the complete sides array with nested
player data — the API response is returned from this column as-is.
Denormalised text columns (teams_csv, player_names_lower, player_mlb_ids_csv)
enable efficient filtering without parsing the JSON at query time.
"""

from sqlalchemy import Column, Integer, Float, String, Boolean, JSON, Text
from app.database import Base


class PastTrade(Base):
    __tablename__ = "past_trades"

    id = Column(Integer, primary_key=True, autoincrement=True)
    trade_id = Column(Integer, unique=True, index=True, nullable=False)
    date = Column(String, index=True, nullable=False)
    year = Column(Integer, index=True, nullable=False)
    description = Column(Text, nullable=True)
    has_cash = Column(Boolean, default=False)
    has_ptbnl = Column(Boolean, default=False)
    n_teams = Column(Integer, default=2)
    n_players = Column(Integer, default=0)

    # Winner / loser (may be overwritten by projection augmentation)
    winner = Column(String, nullable=True)
    winner_name = Column(String, nullable=True)
    loser = Column(String, nullable=True)
    loser_name = Column(String, nullable=True)

    # Aggregate metrics
    surplus_diff = Column(Float, default=0)
    total_trade_war = Column(Float, default=0)
    max_prospect_fv = Column(Integer, nullable=True)

    # Projection augmentation fields
    evaluation_type = Column(String, default="actual")
    projected_winner = Column(String, nullable=True)
    projected_winner_name = Column(String, nullable=True)
    projected_loser = Column(String, nullable=True)
    projected_loser_name = Column(String, nullable=True)
    projected_surplus_diff = Column(Float, nullable=True)
    projected_total_war = Column(Float, nullable=True)

    # Full trade payload (sides → players with all stats)
    sides_json = Column(JSON, default=list)

    # Denormalised columns for fast filtering
    teams_csv = Column(String, index=True)          # "LAD,NYM"
    player_names_lower = Column(Text)                # "juan soto,trea turner,..."
    player_mlb_ids_csv = Column(Text)                # "665742,624413,..."

    def __repr__(self) -> str:
        return f"<PastTrade {self.trade_id} ({self.date})>"

    def to_summary_dict(self) -> dict:
        """Dict for the list endpoint (summaries + sides_json)."""
        d = {
            "trade_id": self.trade_id,
            "date": self.date,
            "year": self.year,
            "description": self.description,
            "has_cash": self.has_cash,
            "has_ptbnl": self.has_ptbnl,
            "n_teams": self.n_teams,
            "n_players": self.n_players,
            "winner": self.winner,
            "winner_name": self.winner_name,
            "loser": self.loser,
            "loser_name": self.loser_name,
            "surplus_diff": self.surplus_diff,
            "total_trade_war": self.total_trade_war,
            "max_prospect_fv": self.max_prospect_fv,
            "evaluation_type": self.evaluation_type,
            "sides": self._build_sides_summary(),
        }
        if self.evaluation_type == "projected":
            d["projected_total_war"] = self.projected_total_war or 0
            d["projected_surplus_diff"] = self.projected_surplus_diff or 0
        return d

    def to_full_dict(self) -> dict:
        """Complete dict for the detail endpoint (all fields + full sides)."""
        d = {
            "trade_id": self.trade_id,
            "date": self.date,
            "year": self.year,
            "description": self.description,
            "has_cash": self.has_cash,
            "has_ptbnl": self.has_ptbnl,
            "n_teams": self.n_teams,
            "n_players": self.n_players,
            "winner": self.winner,
            "winner_name": self.winner_name,
            "loser": self.loser,
            "loser_name": self.loser_name,
            "surplus_diff": self.surplus_diff,
            "total_trade_war": self.total_trade_war,
            "max_prospect_fv": self.max_prospect_fv,
            "evaluation_type": self.evaluation_type,
            "sides": self.sides_json or [],
        }
        if self.evaluation_type == "projected":
            d["projected_winner"] = self.projected_winner
            d["projected_winner_name"] = self.projected_winner_name
            d["projected_loser"] = self.projected_loser
            d["projected_loser_name"] = self.projected_loser_name
            d["projected_surplus_diff"] = self.projected_surplus_diff
            d["projected_total_war"] = self.projected_total_war
        return d

    # ── Internal helpers ──────────────────────────────────────────────────

    def _build_sides_summary(self) -> list:
        """Build the compact sides array for the list endpoint."""
        summaries = []
        for s in (self.sides_json or []):
            players_summary = []
            for p in s.get("players_received", []):
                players_summary.append({
                    "mlb_id": p.get("mlb_id"),
                    "name": p.get("name"),
                    "war_with_team": p.get("war_with_team", 0),
                    "surplus": p.get("surplus", 0),
                    "prospect_fv": p.get("prospect_fv"),
                    "from_team": p.get("from_team"),
                    "from_team_name": p.get("from_team_name"),
                    "projected_war": p.get("projected_war"),
                    "projected_surplus": p.get("projected_surplus"),
                    "has_projection": p.get("has_projection"),
                    "prospect_value": p.get("prospect_value"),
                    "seasons_with_team": p.get("seasons_with_team", 0),
                })
            side_data = {
                "team": s["team"],
                "team_name": s.get("team_name", ""),
                "total_war": s.get("total_war", 0),
                "total_salary": s.get("total_salary", 0),
                "total_war_value": s.get("total_war_value", 0),
                "total_surplus": s.get("total_surplus", 0),
                "players_received": players_summary,
            }
            if "projected_total_war" in s:
                side_data["projected_total_war"] = s["projected_total_war"]
                side_data["projected_total_surplus"] = s.get("projected_total_surplus", 0)
            summaries.append(side_data)
        return summaries
