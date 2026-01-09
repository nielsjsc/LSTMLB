#!/usr/bin/env python3
"""
Static Comparison Report Generator
Generates HTML reports comparing projections without requiring Dash server.
Useful for quick analysis and sharing.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import argparse

# Paths
ROOT_DIR = Path(__file__).parent.parent.parent
DATA_DIR = ROOT_DIR / 'data'
GENERATED_DIR = DATA_DIR / 'generated'
PIPELINE_DIR = GENERATED_DIR / 'pipeline'
OUTPUT_DIR = Path(__file__).parent / 'reports'

# Stats to include in reports
OFFENSIVE_STATS = ['wOBA', 'wRC+', 'OBP', 'SLG', 'AVG', 'BB%', 'K%']
COUNTING_STATS = ['HR', '2B', 'RBI', 'R', 'SB', 'CS', 'PA', 'G']
WAR_COMPONENTS = ['Off', 'BsR', 'Fld', 'Pos', 'Def', 'WAR']
ALL_BATTER_STATS = OFFENSIVE_STATS + COUNTING_STATS + WAR_COMPONENTS

PITCHER_STATS = ['FIP', 'SIERA', 'ERA', 'K%', 'BB%', 'IP', 'G', 'GS', 'WAR']


def load_data():
    """Load prediction datasets"""
    # Load batter predictions with WAR
    pipeline = pd.read_csv(PIPELINE_DIR / 'batter_predictions_with_war.csv')
    
    try:
        notebook = pd.read_csv(GENERATED_DIR / 'batter_predictions_with_war.csv')
    except FileNotFoundError:
        print("Warning: Notebook predictions not found, using pipeline only")
        notebook = pipeline.copy()
    
    # Load pitcher predictions
    try:
        pitcher_pipeline = pd.read_csv(PIPELINE_DIR / 'pitcher_predictions.csv')
        print(f"Loaded {len(pitcher_pipeline)} pitcher predictions")
    except FileNotFoundError:
        print("Warning: Pitcher predictions not found")
        pitcher_pipeline = pd.DataFrame()
    
    # Load historical data
    historical = pd.read_csv(DATA_DIR / 'historic_mlb' / 'mlb_batting_data_1950_2025.csv')
    
    notebook['Source'] = 'Notebook'
    pipeline['Source'] = 'Pipeline'
    
    if 'Season' in historical.columns:
        historical = historical.rename(columns={'Season': 'Year'})
    
    return notebook, pipeline, historical, pitcher_pipeline


def create_player_report(player_id, notebook, pipeline, historical, output_path=None):
    """Generate a comprehensive comparison report for a single player"""
    
    nb_player = notebook[notebook['IDfg'] == player_id].sort_values('Year')
    pl_player = pipeline[pipeline['IDfg'] == player_id].sort_values('Year')
    hist_player = historical[historical['IDfg'] == player_id].sort_values('Year')
    
    if len(nb_player) == 0:
        print(f"Player {player_id} not found in notebook predictions")
        return None
    
    player_name = nb_player['Name'].iloc[0]
    print(f"Generating report for {player_name}...")
    
    # Create multi-panel figure
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            f'wOBA Trajectory', 'wRC+ Trajectory',
            'OBP Trajectory', 'SLG Trajectory',
            'Model Difference (Notebook - Pipeline)', 'Projection Summary'
        ),
        vertical_spacing=0.1,
        horizontal_spacing=0.08
    )
    
    # Color scheme
    colors = {
        'historical': 'white',
        'notebook': '#00d4ff',
        'pipeline': '#ff6b6b'
    }
    
    # Stats to plot
    stats_grid = [
        ('wOBA', 1, 1), ('wRC+', 1, 2),
        ('OBP', 2, 1), ('SLG', 2, 2)
    ]
    
    for stat, row, col in stats_grid:
        if stat not in nb_player.columns:
            continue
            
        # Historical
        if len(hist_player) > 0 and stat in hist_player.columns:
            fig.add_trace(go.Scatter(
                x=hist_player['Year'], y=hist_player[stat],
                mode='lines+markers', name='Historical',
                line=dict(color=colors['historical'], width=2),
                marker=dict(size=6),
                showlegend=(row == 1 and col == 1)
            ), row=row, col=col)
        
        # Notebook
        fig.add_trace(go.Scatter(
            x=nb_player['Year'], y=nb_player[stat],
            mode='lines+markers', name='Notebook',
            line=dict(color=colors['notebook'], width=2),
            marker=dict(size=6, symbol='diamond'),
            showlegend=(row == 1 and col == 1)
        ), row=row, col=col)
        
        # Pipeline
        fig.add_trace(go.Scatter(
            x=pl_player['Year'], y=pl_player[stat],
            mode='lines+markers', name='Pipeline',
            line=dict(color=colors['pipeline'], width=2),
            marker=dict(size=6, symbol='square'),
            showlegend=(row == 1 and col == 1)
        ), row=row, col=col)
    
    # Difference plot (row 3, col 1)
    if 'wOBA' in nb_player.columns and 'wOBA' in pl_player.columns:
        merged = pd.merge(
            nb_player[['Year', 'wOBA']].rename(columns={'wOBA': 'NB'}),
            pl_player[['Year', 'wOBA']].rename(columns={'wOBA': 'PL'}),
            on='Year'
        )
        merged['Diff'] = merged['NB'] - merged['PL']
        
        bar_colors = ['#00d4ff' if d >= 0 else '#ff6b6b' for d in merged['Diff']]
        
        fig.add_trace(go.Bar(
            x=merged['Year'], y=merged['Diff'],
            marker_color=bar_colors,
            name='wOBA Diff',
            showlegend=False
        ), row=3, col=1)
    
    # Summary stats (row 3, col 2)
    summary_years = [2026, 2028, 2030]
    summary_text = "<b>Projection Summary</b><br><br>"
    
    for year in summary_years:
        nb_yr = nb_player[nb_player['Year'] == year]
        pl_yr = pl_player[pl_player['Year'] == year]
        
        if len(nb_yr) > 0 and len(pl_yr) > 0:
            summary_text += f"<b>{year}:</b><br>"
            for stat in ['wOBA', 'wRC+', 'HR']:
                if stat in nb_yr.columns and stat in pl_yr.columns:
                    nb_val = nb_yr[stat].iloc[0]
                    pl_val = pl_yr[stat].iloc[0]
                    diff = nb_val - pl_val
                    sign = '+' if diff >= 0 else ''
                    summary_text += f"  {stat}: NB={nb_val:.3f}, PL={pl_val:.3f} ({sign}{diff:.3f})<br>"
            summary_text += "<br>"
    
    fig.add_annotation(
        x=0.5, y=0.5,
        text=summary_text,
        showarrow=False,
        font=dict(size=11),
        xref="x6 domain", yref="y6 domain",
        align="left"
    )
    
    # Layout
    fig.update_layout(
        title=dict(
            text=f"<b>{player_name}</b> - Projection Comparison Report",
            font=dict(size=20)
        ),
        template="plotly_dark",
        height=900,
        width=1200,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    # Save report
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    if output_path is None:
        safe_name = player_name.replace(' ', '_').replace('.', '')
        output_path = OUTPUT_DIR / f"{safe_name}_comparison.html"
    
    fig.write_html(str(output_path), include_plotlyjs='cdn')
    print(f"Report saved to: {output_path}")
    
    return fig


def create_summary_report(notebook, pipeline, historical, top_n=20):
    """Create a summary report comparing top players"""
    
    print("Generating summary report for top players...")
    
    # Get top players by 2026 WAR in notebook predictions
    nb_2026 = notebook[notebook['Year'] == 2026].nlargest(top_n, 'wOBA')
    
    # Create comparison dataframe
    comparison_data = []
    
    for _, row in nb_2026.iterrows():
        player_id = row['IDfg']
        player_name = row['Name']
        
        nb_player = notebook[(notebook['IDfg'] == player_id) & (notebook['Year'] == 2026)]
        pl_player = pipeline[(pipeline['IDfg'] == player_id) & (pipeline['Year'] == 2026)]
        
        if len(nb_player) == 0 or len(pl_player) == 0:
            continue
        
        nb_woba = nb_player['wOBA'].iloc[0]
        pl_woba = pl_player['wOBA'].iloc[0]
        
        nb_wrc = nb_player['wRC+'].iloc[0] if 'wRC+' in nb_player.columns else np.nan
        pl_wrc = pl_player['wRC+'].iloc[0] if 'wRC+' in pl_player.columns else np.nan
        
        comparison_data.append({
            'Name': player_name,
            'Age': nb_player['Age'].iloc[0],
            'NB_wOBA': nb_woba,
            'PL_wOBA': pl_woba,
            'wOBA_Diff': nb_woba - pl_woba,
            'NB_wRC+': nb_wrc,
            'PL_wRC+': pl_wrc,
            'wRC+_Diff': nb_wrc - pl_wrc if not np.isnan(nb_wrc) else np.nan
        })
    
    comp_df = pd.DataFrame(comparison_data)
    
    # Create visualization
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('wOBA Comparison (2026)', 'wRC+ Comparison (2026)'),
        vertical_spacing=0.15
    )
    
    # wOBA comparison
    fig.add_trace(go.Bar(
        x=comp_df['Name'], y=comp_df['NB_wOBA'],
        name='Notebook', marker_color='#00d4ff'
    ), row=1, col=1)
    
    fig.add_trace(go.Bar(
        x=comp_df['Name'], y=comp_df['PL_wOBA'],
        name='Pipeline', marker_color='#ff6b6b'
    ), row=1, col=1)
    
    # wRC+ comparison
    fig.add_trace(go.Bar(
        x=comp_df['Name'], y=comp_df['NB_wRC+'],
        name='Notebook', marker_color='#00d4ff', showlegend=False
    ), row=2, col=1)
    
    fig.add_trace(go.Bar(
        x=comp_df['Name'], y=comp_df['PL_wRC+'],
        name='Pipeline', marker_color='#ff6b6b', showlegend=False
    ), row=2, col=1)
    
    fig.update_layout(
        title="<b>Top Players: Model Comparison (2026 Projections)</b>",
        template="plotly_dark",
        height=800,
        width=1400,
        barmode='group',
        legend=dict(orientation="h", yanchor="bottom", y=1.02)
    )
    
    fig.update_xaxes(tickangle=45)
    
    OUTPUT_DIR.mkdir(exist_ok=True)
    output_path = OUTPUT_DIR / "summary_comparison.html"
    fig.write_html(str(output_path), include_plotlyjs='cdn')
    
    # Also save CSV
    csv_path = OUTPUT_DIR / "model_comparison.csv"
    comp_df.to_csv(csv_path, index=False)
    
    print(f"Summary report saved to: {output_path}")
    print(f"Comparison data saved to: {csv_path}")
    
    return fig, comp_df


def main():
    parser = argparse.ArgumentParser(description='Generate projection comparison reports')
    parser.add_argument('--player', type=str, help='Player name or ID to generate report for')
    parser.add_argument('--summary', action='store_true', help='Generate summary report for top players')
    parser.add_argument('--all', action='store_true', help='Generate reports for all common players')
    args = parser.parse_args()
    
    notebook, pipeline, historical, pitcher_pipeline = load_data()
    
    if args.summary or (not args.player and not args.all):
        create_summary_report(notebook, pipeline, historical)
    
    if args.player:
        # Try to find player by name or ID
        if args.player.isdigit():
            player_id = int(args.player)
        else:
            matches = notebook[notebook['Name'].str.contains(args.player, case=False, na=False)]
            if len(matches) == 0:
                print(f"No players found matching '{args.player}'")
                return
            player_id = matches['IDfg'].iloc[0]
            print(f"Found: {matches['Name'].iloc[0]} (IDfg: {player_id})")
        
        create_player_report(player_id, notebook, pipeline, historical)
    
    if args.all:
        common_ids = set(notebook['IDfg']).intersection(set(pipeline['IDfg']))
        print(f"Generating reports for {len(common_ids)} players...")
        for pid in common_ids:
            try:
                create_player_report(pid, notebook, pipeline, historical)
            except Exception as e:
                print(f"Error for player {pid}: {e}")


if __name__ == "__main__":
    main()
