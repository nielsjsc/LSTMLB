#!/usr/bin/env python3
"""
Interactive Projection Comparison Tool
Compares batter projections between notebook and auto_train pipeline models.
Focuses on offensive statistics and career trajectory analysis.

Author: Niels Christoffersen
"""

import pandas as pd
import numpy as np
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from dash import Dash, dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc

# Paths
ROOT_DIR = Path(__file__).parent.parent.parent
DATA_DIR = ROOT_DIR / 'data'
GENERATED_DIR = DATA_DIR / 'generated'
PIPELINE_DIR = GENERATED_DIR / 'pipeline'

# Offensive stats to compare (all available in batter_predictions_with_war.csv)
OFFENSIVE_STATS = ['wRC+', 'wOBA' , 'OBP', 'SLG', 'AVG', 'BB%', 'K%']
COUNTING_STATS = ['HR', '2B', 'RBI', 'R', 'SB', 'CS', 'PA', 'G']
WAR_COMPONENTS = ['Off', 'BsR', 'Fld', 'Pos', 'Def', 'WAR']

# All stats to display in comparison
ALL_BATTER_STATS = OFFENSIVE_STATS + COUNTING_STATS + WAR_COMPONENTS

# Pitcher stats to compare
PITCHER_STATS = ['FIP', 'SIERA', 'ERA', 'K%', 'BB%', 'IP', 'G', 'GS', 'WAR']


def load_data():
    """Load all required datasets"""
    print("Loading data...")
    
    # Load batting predictions with WAR from pipeline
    pipeline_preds = pd.read_csv(PIPELINE_DIR / 'batter_predictions_with_war.csv')
    
    # For notebook comparison, check if it exists, otherwise use pipeline as baseline
    try:
        notebook_preds = pd.read_csv(GENERATED_DIR / 'batter_predictions_with_war.csv')
        print("Loaded notebook predictions with WAR")
    except FileNotFoundError:
        print("Warning: Notebook predictions not found, using pipeline as single source")
        notebook_preds = pipeline_preds.copy()
    
    # Load pitcher predictions
    try:
        pipeline_pitchers = pd.read_csv(PIPELINE_DIR / 'pitcher_predictions.csv')
        print(f"Loaded {len(pipeline_pitchers)} pipeline pitcher predictions")
    except FileNotFoundError:
        print("Warning: Pipeline pitcher predictions not found")
        pipeline_pitchers = pd.DataFrame()
    
    try:
        notebook_pitchers = pd.read_csv(GENERATED_DIR / 'pitcher_predictions.csv')
        print(f"Loaded {len(notebook_pitchers)} notebook pitcher predictions")
    except FileNotFoundError:
        print("Warning: Notebook pitcher predictions not found, using pipeline as single source")
        notebook_pitchers = pipeline_pitchers.copy()
    
    # Load historical data
    historical = pd.read_csv(DATA_DIR / 'historic_mlb' / 'mlb_batting_data_1950_2025_with_statcast.csv')
    historical_pitchers = pd.read_csv(DATA_DIR / 'historic_mlb' / 'mlb_pitching_data_1950_2025_with_statcast.csv')
    
    # Tag sources
    notebook_preds['Source'] = 'Notebook'
    pipeline_preds['Source'] = 'Pipeline'
    notebook_pitchers['Source'] = 'Notebook'
    pipeline_pitchers['Source'] = 'Pipeline'
    
    # Standardize column names
    if 'Season' in historical.columns:
        historical = historical.rename(columns={'Season': 'Year'})
    if 'Season' in historical_pitchers.columns:
        historical_pitchers = historical_pitchers.rename(columns={'Season': 'Year'})
    if 'Season' in notebook_pitchers.columns:
        notebook_pitchers = notebook_pitchers.rename(columns={'Season': 'Year'})
    if 'Season' in pipeline_pitchers.columns:
        pipeline_pitchers = pipeline_pitchers.rename(columns={'Season': 'Year'})
    
    print(f"Loaded {len(notebook_preds)} notebook batter predictions")
    print(f"Loaded {len(pipeline_preds)} pipeline batter predictions")
    print(f"Loaded {len(historical)} historical batter records")
    print(f"Loaded {len(historical_pitchers)} historical pitcher records")
    
    return notebook_preds, pipeline_preds, historical, notebook_pitchers, pipeline_pitchers, historical_pitchers


def get_player_list(notebook_preds, pipeline_preds, notebook_pitchers, pipeline_pitchers):
    """Get list of players that appear in both prediction sets (batters and pitchers)"""
    # Get common batters
    notebook_batters = set(notebook_preds['IDfg'].unique())
    pipeline_batters = set(pipeline_preds['IDfg'].unique())
    common_batters = notebook_batters.intersection(pipeline_batters)
    
    # Get common pitchers
    notebook_pitcher_ids = set(notebook_pitchers['IDfg'].unique()) if not notebook_pitchers.empty else set()
    pipeline_pitcher_ids = set(pipeline_pitchers['IDfg'].unique()) if not pipeline_pitchers.empty else set()
    common_pitchers = notebook_pitcher_ids.intersection(pipeline_pitcher_ids)
    
    # Create name->IDfg mapping for batters
    all_batters = pd.concat([notebook_preds, pipeline_preds])
    cols = ['Name', 'IDfg']
    if 'Position' in all_batters.columns:
        cols.append('Position')
    if 'Team' in all_batters.columns:
        cols.append('Team')
    batter_names = all_batters[all_batters['IDfg'].isin(common_batters)][cols].drop_duplicates(subset=['IDfg'])
    batter_names['PlayerType'] = 'Batter'
    
    # Create name->IDfg mapping for pitchers
    pitcher_names = pd.DataFrame()
    if not pipeline_pitchers.empty:
        all_pitchers = pd.concat([notebook_pitchers, pipeline_pitchers])
        pitcher_cols = ['Name', 'IDfg']
        if 'Team' in all_pitchers.columns:
            pitcher_cols.append('Team')
        pitcher_names = all_pitchers[all_pitchers['IDfg'].isin(common_pitchers)][pitcher_cols].drop_duplicates(subset=['IDfg'])
        pitcher_names['PlayerType'] = 'Pitcher'
        if 'Position' not in pitcher_cols and 'Position' in cols:
            pitcher_names['Position'] = 'P'
    
    # Combine and sort
    player_names = pd.concat([batter_names, pitcher_names], ignore_index=True)
    player_names = player_names.sort_values('Name')
    
    return player_names


def create_app(notebook_preds, pipeline_preds, historical, notebook_pitchers, pipeline_pitchers, historical_pitchers):
    """Create the Dash application"""
    
    player_list = get_player_list(notebook_preds, pipeline_preds, notebook_pitchers, pipeline_pitchers)
    player_options = [{'label': row['Name'], 'value': row['IDfg']} 
                      for _, row in player_list.iterrows()]
    
    # Get available stats (all batter stats and pitcher stats)
    available_batter_stats = [s for s in ALL_BATTER_STATS if s in notebook_preds.columns and s in pipeline_preds.columns]
    available_pitcher_stats = [s for s in PITCHER_STATS if s in pipeline_pitchers.columns] if not pipeline_pitchers.empty else []
    
    app = Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
    
    app.layout = dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H1("⚾ Player Projection Comparison", className="text-center my-4"),
                html.P("Compare offensive and defensive career trajectories between Notebook and Pipeline models", 
                       className="text-center text-muted")
            ])
        ]),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Player Selection"),
                    dbc.CardBody([
                        dcc.Dropdown(
                            id='player-dropdown',
                            options=player_options,
                            value=player_options[0]['value'] if player_options else None,
                            placeholder="Select a player...",
                            className="mb-3",
                            style={'color': 'black'}
                        ),
                        dcc.Dropdown(
                            id='stat-dropdown',
                            options=[{'label': s, 'value': s} for s in available_batter_stats],
                            value='WAR',
                            placeholder="Select stat to compare...",
                            className="mb-3",
                            style={'color': 'black'}
                        ),
                        dbc.Checklist(
                            id='show-historical',
                            options=[{'label': 'Show Historical Data', 'value': 'show'}],
                            value=['show'],
                            switch=True
                        )
                    ])
                ], className="mb-4")
            ], width=12)
        ]),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Career Trajectory Comparison"),
                    dbc.CardBody([
                        dcc.Graph(id='trajectory-plot', style={'height': '500px'})
                    ])
                ])
            ], width=12)
        ], className="mb-4"),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Multi-Stat Comparison"),
                    dbc.CardBody([
                        dcc.Graph(id='multistat-plot', style={'height': '400px'})
                    ])
                ])
            ], width=6),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Projection Difference Over Time"),
                    dbc.CardBody([
                        dcc.Graph(id='difference-plot', style={'height': '400px'})
                    ])
                ])
            ], width=6)
        ], className="mb-4"),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Counting Stats Comparison"),
                    dbc.CardBody([
                        dcc.Graph(id='counting-plot', style={'height': '350px'})
                    ])
                ])
            ], width=12)
        ], className="mb-4"),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Side-by-Side Data Table"),
                    dbc.CardBody([
                        html.Div(id='comparison-table')
                    ])
                ])
            ], width=12)
        ])
        
    ], fluid=True)
    
    # Callback to update stat dropdown based on player type
    @app.callback(
        [Output('stat-dropdown', 'options'),
         Output('stat-dropdown', 'value')],
        [Input('player-dropdown', 'value')]
    )
    def update_stat_dropdown(player_id):
        if not player_id:
            return [{'label': s, 'value': s} for s in available_batter_stats], 'WAR'
        
        # Determine player type
        player_info = player_list[player_list['IDfg'] == player_id]
        if player_info.empty:
            return [{'label': s, 'value': s} for s in available_batter_stats], 'WAR'
        
        player_type = player_info['PlayerType'].iloc[0]
        
        if player_type == 'Pitcher':
            options = [{'label': s, 'value': s} for s in available_pitcher_stats]
            default_value = 'WAR' if 'WAR' in available_pitcher_stats else (available_pitcher_stats[0] if available_pitcher_stats else None)
        else:
            options = [{'label': s, 'value': s} for s in available_batter_stats]
            default_value = 'WAR'
        
        return options, default_value
    
    @app.callback(
        [Output('trajectory-plot', 'figure'),
         Output('multistat-plot', 'figure'),
         Output('difference-plot', 'figure'),
         Output('counting-plot', 'figure'),
         Output('comparison-table', 'children')],
        [Input('player-dropdown', 'value'),
         Input('stat-dropdown', 'value'),
         Input('show-historical', 'value')]
    )
    def update_plots(player_id, selected_stat, show_hist):
        if not player_id:
            empty_fig = go.Figure()
            return empty_fig, empty_fig, empty_fig, empty_fig, html.Div()
        
        # Determine player type
        player_info = player_list[player_list['IDfg'] == player_id]
        if player_info.empty:
            empty_fig = go.Figure()
            return empty_fig, empty_fig, empty_fig, empty_fig, html.Div()
        
        player_type = player_info['PlayerType'].iloc[0]
        
        # Filter data for selected player based on type
        if player_type == 'Pitcher':
            nb_player = notebook_pitchers[notebook_pitchers['IDfg'] == player_id].copy()
            pl_player = pipeline_pitchers[pipeline_pitchers['IDfg'] == player_id].copy()
            hist_player = historical_pitchers[historical_pitchers['IDfg'] == player_id].copy()
        else:  # Batter
            nb_player = notebook_preds[notebook_preds['IDfg'] == player_id].copy()
            pl_player = pipeline_preds[pipeline_preds['IDfg'] == player_id].copy()
            hist_player = historical[historical['IDfg'] == player_id].copy()
        
        player_name = nb_player['Name'].iloc[0] if len(nb_player) > 0 else "Unknown"
        
        # 1. Career Trajectory Plot
        trajectory_fig = create_trajectory_plot(
            nb_player, pl_player, hist_player, 
            selected_stat, player_name, 'show' in (show_hist or [])
        )
        
        # Use appropriate stats based on player type
        stats_to_use = available_pitcher_stats if player_type == 'Pitcher' else available_batter_stats
        
        # 2. Multi-stat radar comparison
        multistat_fig = create_multistat_plot(nb_player, pl_player, player_name, stats_to_use)
        
        # 3. Difference over time
        difference_fig = create_difference_plot(nb_player, pl_player, selected_stat, player_name)
        
        # 4. Counting stats comparison
        counting_fig = create_counting_plot(nb_player, pl_player, player_name, player_type)
        
        # 5. Data table
        table = create_comparison_table(nb_player, pl_player, stats_to_use)
        
        return trajectory_fig, multistat_fig, difference_fig, counting_fig, table
    
    return app


def create_trajectory_plot(nb_data, pl_data, hist_data, stat, player_name, show_historical):
    """Create the main career trajectory comparison plot"""
    fig = go.Figure()
    
    # Historical data
    if show_historical and len(hist_data) > 0 and stat in hist_data.columns:
        hist_data = hist_data.sort_values('Year')
        fig.add_trace(go.Scatter(
            x=hist_data['Year'],
            y=hist_data[stat],
            mode='lines+markers',
            name='Historical',
            line=dict(color='white', width=3),
            marker=dict(size=10, symbol='circle')
        ))
    
    # Notebook predictions
    if len(nb_data) > 0 and stat in nb_data.columns:
        nb_data = nb_data.sort_values('Year')
        fig.add_trace(go.Scatter(
            x=nb_data['Year'],
            y=nb_data[stat],
            mode='lines+markers',
            name='Notebook Model',
            line=dict(color='#00d4ff', width=2, dash='solid'),
            marker=dict(size=8, symbol='diamond')
        ))
    
    # Pipeline predictions
    if len(pl_data) > 0 and stat in pl_data.columns:
        pl_data = pl_data.sort_values('Year')
        fig.add_trace(go.Scatter(
            x=pl_data['Year'],
            y=pl_data[stat],
            mode='lines+markers',
            name='Pipeline Model',
            line=dict(color='#ff6b6b', width=2, dash='solid'),
            marker=dict(size=8, symbol='square')
        ))
    
    # Add vertical line at projection start
    if show_historical and len(hist_data) > 0:
        last_historical_year = hist_data['Year'].max()
        fig.add_vline(
            x=last_historical_year + 0.5, 
            line_dash="dash", 
            line_color="gray",
            annotation_text="Projections →",
            annotation_position="top right"
        )
    
    fig.update_layout(
        title=f"{player_name}: {stat} Career Trajectory",
        xaxis_title="Year",
        yaxis_title=stat,
        template="plotly_dark",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig


def create_multistat_plot(nb_data, pl_data, player_name, stats):
    """Create radar chart comparing multiple stats for a single year"""
    # Use 2026 (first projection year) for comparison
    nb_2026 = nb_data[nb_data['Year'] == 2026]
    pl_2026 = pl_data[pl_data['Year'] == 2026]
    
    if len(nb_2026) == 0 or len(pl_2026) == 0:
        return go.Figure().update_layout(
            title="No 2026 data available",
            template="plotly_dark"
        )
    
    # Normalize stats for radar chart
    fig = go.Figure()
    
    # Get available stats that exist in both
    available = [s for s in stats if s in nb_2026.columns and s in pl_2026.columns]
    
    nb_values = [nb_2026[s].iloc[0] for s in available]
    pl_values = [pl_2026[s].iloc[0] for s in available]
    
    # Normalize to 0-1 scale for comparison
    all_vals = nb_values + pl_values
    min_val = min(all_vals) if all_vals else 0
    max_val = max(all_vals) if all_vals else 1
    range_val = max_val - min_val if max_val != min_val else 1
    
    nb_norm = [(v - min_val) / range_val for v in nb_values]
    pl_norm = [(v - min_val) / range_val for v in pl_values]
    
    fig.add_trace(go.Scatterpolar(
        r=nb_norm + [nb_norm[0]],  # Close the polygon
        theta=available + [available[0]],
        fill='toself',
        name='Notebook',
        line_color='#00d4ff',
        fillcolor='rgba(0, 212, 255, 0.3)'
    ))
    
    fig.add_trace(go.Scatterpolar(
        r=pl_norm + [pl_norm[0]],
        theta=available + [available[0]],
        fill='toself',
        name='Pipeline',
        line_color='#ff6b6b',
        fillcolor='rgba(255, 107, 107, 0.3)'
    ))
    
    fig.update_layout(
        title=f"{player_name}: 2026 Stats Comparison",
        template="plotly_dark",
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1]),
            bgcolor='rgba(0,0,0,0)'
        ),
        showlegend=True
    )
    
    return fig


def create_difference_plot(nb_data, pl_data, stat, player_name):
    """Create plot showing difference between models over time"""
    if stat not in nb_data.columns or stat not in pl_data.columns:
        return go.Figure().update_layout(title="Stat not available", template="plotly_dark")
    
    # Merge on year
    merged = pd.merge(
        nb_data[['Year', stat]].rename(columns={stat: 'Notebook'}),
        pl_data[['Year', stat]].rename(columns={stat: 'Pipeline'}),
        on='Year',
        how='inner'
    )
    
    if len(merged) == 0:
        return go.Figure().update_layout(title="No overlapping years", template="plotly_dark")
    
    merged['Difference'] = merged['Notebook'] - merged['Pipeline']
    merged['Pct_Diff'] = (merged['Difference'] / merged['Pipeline']) * 100
    
    fig = go.Figure()
    
    # Bar chart of differences
    colors = ['#00d4ff' if d >= 0 else '#ff6b6b' for d in merged['Difference']]
    
    fig.add_trace(go.Bar(
        x=merged['Year'],
        y=merged['Difference'],
        marker_color=colors,
        name='Difference (Notebook - Pipeline)',
        hovertemplate='Year: %{x}<br>Difference: %{y:.3f}<extra></extra>'
    ))
    
    fig.add_hline(y=0, line_dash="dash", line_color="white")
    
    fig.update_layout(
        title=f"{player_name}: {stat} Difference (Notebook - Pipeline)",
        xaxis_title="Year",
        yaxis_title=f"{stat} Difference",
        template="plotly_dark"
    )
    
    return fig


def create_counting_plot(nb_data, pl_data, player_name, player_type='Batter'):
    """Create grouped bar chart for counting stats"""
    # Use 2026 for comparison
    nb_2026 = nb_data[nb_data['Year'] == 2026]
    pl_2026 = pl_data[pl_data['Year'] == 2026]
    
    if len(nb_2026) == 0 or len(pl_2026) == 0:
        return go.Figure().update_layout(title="No 2026 data", template="plotly_dark")
    
    # Use appropriate counting stats based on player type
    if player_type == 'Pitcher':
        counting_stats = ['IP', 'G', 'GS']
    else:
        counting_stats = COUNTING_STATS
    
    available_counting = [s for s in counting_stats if s in nb_2026.columns and s in pl_2026.columns]
    
    if not available_counting:
        return go.Figure().update_layout(title="No counting stats available", template="plotly_dark")
    
    nb_vals = [nb_2026[s].iloc[0] for s in available_counting]
    pl_vals = [pl_2026[s].iloc[0] for s in available_counting]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Notebook',
        x=available_counting,
        y=nb_vals,
        marker_color='#00d4ff'
    ))
    
    fig.add_trace(go.Bar(
        name='Pipeline',
        x=available_counting,
        y=pl_vals,
        marker_color='#ff6b6b'
    ))
    
    fig.update_layout(
        title=f"{player_name}: 2026 Counting Stats",
        xaxis_title="Stat",
        yaxis_title="Value",
        template="plotly_dark",
        barmode='group'
    )
    
    return fig


def create_comparison_table(nb_data, pl_data, stats):
    """Create side-by-side comparison table"""
    # Get common years
    common_years = set(nb_data['Year']).intersection(set(pl_data['Year']))
    
    if not common_years:
        return html.Div("No common years to compare")
    
    rows = []
    for year in sorted(common_years)[:5]:  # Show first 5 years
        nb_row = nb_data[nb_data['Year'] == year]
        pl_row = pl_data[pl_data['Year'] == year]
        
        if len(nb_row) == 0 or len(pl_row) == 0:
            continue
        
        row_data = {'Year': int(year), 'Age': int(nb_row['Age'].iloc[0])}
        
        for stat in stats[:5]:  # Top 5 stats
            if stat in nb_row.columns and stat in pl_row.columns:
                nb_val = nb_row[stat].iloc[0]
                pl_val = pl_row[stat].iloc[0]
                row_data[f'{stat} (NB)'] = f"{nb_val:.3f}" if isinstance(nb_val, float) else nb_val
                row_data[f'{stat} (PL)'] = f"{pl_val:.3f}" if isinstance(pl_val, float) else pl_val
        
        rows.append(row_data)
    
    if not rows:
        return html.Div("No data to display")
    
    df = pd.DataFrame(rows)
    
    table = dbc.Table.from_dataframe(
        df, 
        striped=True, 
        bordered=True, 
        hover=True,
        color='dark',
        size='sm'
    )
    
    return table


def main():
    """Main entry point"""
    print("=" * 60)
    print("Batter Projection Comparison Tool")
    print("=" * 60)
    
    # Load data
    notebook_preds, pipeline_preds, historical, notebook_pitchers, pipeline_pitchers, historical_pitchers = load_data()
    
    # Create and run app
    app = create_app(notebook_preds, pipeline_preds, historical, notebook_pitchers, pipeline_pitchers, historical_pitchers)
    
    print("\nStarting server...")
    print("Open http://127.0.0.1:8050 in your browser")
    print("Press Ctrl+C to quit\n")
    
    app.run(debug=True, port=8050)


if __name__ == "__main__":
    main()
