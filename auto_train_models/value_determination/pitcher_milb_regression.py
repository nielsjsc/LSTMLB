import pandas as pd
import numpy as np

# ---------------------------------------------------------
# 1. CONFIGURATION & CONSTANTS
# ---------------------------------------------------------

# Stabilization thresholds (measured in Total Batters Faced - TBF)
# Pitching metrics stabilize at vastly different speeds.
STABILIZATION_TBF = {
    'K%': 70,       # Stabilizes very quickly
    'BB%': 120,     # Stabilizes quickly
    'GB%': 70,      # (Measured in BIP, but using TBF proxy for simplicity)
    'HR/FB': 400,   # Takes a long time to stabilize
    'BABIP': 2000   # Pitcher BABIP takes multiple seasons to stabilize
}

# Minor League Equivalency (MLE) Multipliers
# Adjusts MiLB performance to expected MLB baseline based on level difficulty
MLE_LEVEL_FACTORS = {
    'AAA': {'K%': 0.85, 'BB%': 1.10, 'GB%': 1.00, 'HR/FB': 1.15, 'BABIP': 1.05},
    'AA':  {'K%': 0.78, 'BB%': 1.15, 'GB%': 1.00, 'HR/FB': 1.20, 'BABIP': 1.08},
    'A+':  {'K%': 0.72, 'BB%': 1.20, 'GB%': 1.00, 'HR/FB': 1.25, 'BABIP': 1.10},
    'A':   {'K%': 0.68, 'BB%': 1.25, 'GB%': 1.00, 'HR/FB': 1.30, 'BABIP': 1.12},
    'A-':  {'K%': 0.65, 'BB%': 1.30, 'GB%': 1.00, 'HR/FB': 1.35, 'BABIP': 1.15},
    'CPX': {'K%': 0.60, 'BB%': 1.40, 'GB%': 1.00, 'HR/FB': 1.45, 'BABIP': 1.20}
}

# ---------------------------------------------------------
# 2. DATA LOADING ABSTRACTIONS
# ---------------------------------------------------------

def _load_career_tbf(idfg, df_mlb_history):
    """Fetches total MLB batters faced for stabilization checks."""
    # If the calling script didn't pass historical data, default to 0 TBF
    if df_mlb_history is None:
        return 0.0
        
    try:
        return df_mlb_history.loc[df_mlb_history['IDfg'] == idfg, 'TBF'].sum()
    except KeyError:
        return 0.0
def _load_milb_data(idfg, df_milb_history):
    """Isolates the most relevant/recent MiLB stint for the pitcher."""
    player_milb = df_milb_history[df_milb_history['IDfg'] == idfg]
    if player_milb.empty:
        return None
    # Sort by Year/Level or apply a weighted average of recent MiLB seasons.
    # The MiLB pitcher export uses 'Season' (like the hitter file), not
    # 'Year' — fall back gracefully instead of assuming either name.
    if 'Year' in player_milb.columns:
        sort_col = 'Year'
    elif 'Season' in player_milb.columns:
        sort_col = 'Season'
    else:
        sort_col = None

    if sort_col is not None:
        player_milb = player_milb.sort_values(by=sort_col, ascending=False)
    return player_milb.iloc[0]

# ---------------------------------------------------------
# 3. MAIN REGRESSION ENGINE
# ---------------------------------------------------------

def apply_pitcher_milb_regression(df_preds, df_milb_history, df_mlb_history=None, role=None, **kwargs):
    """
    Blends LSTM base component projections with MiLB MLEs for 
    pitchers who have not yet reached stabilization thresholds.
    """
    df = df_preds.copy()
    base_components = ['K%', 'BB%', 'GB%', 'HR/FB', 'BABIP']

    # Coerce numeric columns — CSV exports sometimes bring these in as
    # strings (e.g. via mixed-type columns or thousands separators), which
    # silently breaks downstream arithmetic (e.g. "20.5" * 0.85 raises
    # TypeError instead of computing). Coerce on copies so we don't mutate
    # the caller's dataframes.
    df_milb_history = df_milb_history.copy()
    for comp in base_components:
        if comp in df_milb_history.columns:
            df_milb_history[comp] = pd.to_numeric(df_milb_history[comp], errors='coerce')
    for comp in base_components:
        if comp in df.columns:
            df[comp] = pd.to_numeric(df[comp], errors='coerce')
    if df_mlb_history is not None and 'TBF' in df_mlb_history.columns:
        df_mlb_history = df_mlb_history.copy()
        df_mlb_history['TBF'] = pd.to_numeric(df_mlb_history['TBF'], errors='coerce')

    # Iterate through predictions grouped by player to maintain aging curves
    for idfg, group in df.groupby('IDfg'):
        
        # 1. Fetch MLB Sample Size
        mlb_tbf = _load_career_tbf(idfg, df_mlb_history)
        
        # If fully stabilized in the slowest metric (BABIP), keep raw LSTM outputs
        if mlb_tbf >= STABILIZATION_TBF['BABIP']:
            continue
            
        # 2. Fetch Minor League Data
        milb_data = _load_milb_data(idfg, df_milb_history)
        if milb_data is None:
            continue
            
        # 3. Calculate MLE Baseline
        level = milb_data.get('Level', 'AAA')
        factors = MLE_LEVEL_FACTORS.get(level, MLE_LEVEL_FACTORS['AAA'])
        
        mle_baseline = {}
        for comp in base_components:
            # Ensure the MiLB data actually contains the component before math,
            # and that it's a real (non-NaN) value -- blank/unparseable CSV
            # cells coerce to NaN and must fall back the same way a fully
            # missing column does, or NaN silently poisons every downstream
            # stat for this pitcher (FIP, ERA, IP, counting stats...).
            comp_val = milb_data.get(comp) if comp in milb_data else None
            if comp_val is not None and pd.notna(comp_val):
                mle_baseline[comp] = comp_val * factors[comp]
            else:
                mle_baseline[comp] = group[comp].mean() # Fallback to LSTM if missing

        # 4. Apply Component-Specific Bayesian Blend Row-by-Row
        for idx, row in group.iterrows():
            for comp in base_components:
                # Calculate how much weight to give the MLB projection (LSTM)
                stab_threshold = STABILIZATION_TBF[comp]
                
                # MLB weight scales from 0.0 to 1.0 based on career TBF
                mlb_weight = min(mlb_tbf / stab_threshold, 1.0)
                milb_weight = 1.0 - mlb_weight
                
                # Blend the original LSTM prediction with the translated MLE
                blended_value = (row[comp] * mlb_weight) + (mle_baseline[comp] * milb_weight)

                # If either input was NaN (e.g. this pitcher's own LSTM row
                # is missing the stat too), keep the original value instead
                # of writing NaN into the dataframe.
                if pd.notna(blended_value):
                    df.at[idx, comp] = blended_value

    # 5. Reconstruct the Macro Metrics (ERA, FIP, LOB%, K/9)
    df = _reconstruct_pitcher_derived_stats(df)
    
    return df

# ---------------------------------------------------------
# 4. DOWNSTREAM RECONSTRUCTION
# ---------------------------------------------------------


def _reconstruct_pitcher_derived_stats(df, cFIP=3.20, lg_BABIP=0.295, lg_LOB=0.72):
    """
    Rebuilds /9s, FIP, LOB%, and ERA using blended base components.
    Expects df to contain 'K%', 'BB%', 'GB%', 'HR/FB', and 'BABIP'.
    """
    # 1. Fill missing secondary components with MLB averages if not output by LSTM
    if 'HBP%' not in df.columns:
        df['HBP%'] = 0.011 
    if 'FB%' not in df.columns:
        # Assuming ~20% Line Drive rate if FB% isn't natively projected
        df['FB%'] = 1.0 - df['GB%'] - 0.20 
        
    # 2. Establish a baseline TBF to generate proportional counting stats.
    # (If your df already has a 'Projected_TBF', use it. Otherwise, 600 works 
    # perfectly since rates scale proportionally regardless of the absolute volume).
    tbf = df.get('Projected_TBF', 600.0)
    
    # 3. Calculate Absolute Events
    K = df['K%'] * tbf
    BB = df['BB%'] * tbf
    HBP = df['HBP%'] * tbf
    
    # Total balls making contact (excluding strikeouts, walks, and hit-by-pitches)
    contact_events = tbf - K - BB - HBP
    
    FB = contact_events * df['FB%']
    HR = FB * df['HR/FB']
    
    # Actual Balls in Play (BIP) excludes Home Runs
    BIP = contact_events - HR
    Hits = HR + (BIP * df['BABIP'])
    
    # 4. Calculate Outs and Innings Pitched (IP)
    # Outs = Strikeouts + Non-Hit Balls in Play
    outs = K + (BIP * (1.0 - df['BABIP']))
    
    # Avoid Division by Zero for players with weird fractional projections
    IP = outs / 3.0
    IP = IP.replace(0, 0.1) 
    
    # 5. Reconstruct the /9 Metrics
    df['K/9'] = (K / IP) * 9.0
    df['BB/9'] = (BB / IP) * 9.0
    df['HR/9'] = (HR / IP) * 9.0
    df['H/9'] = (Hits / IP) * 9.0
    
    # 6. Reconstruct FIP
    df['FIP'] = ((13.0 * HR) + (3.0 * (BB + HBP)) - (2.0 * K)) / IP + cFIP
    
    # 7. Reconstruct LOB% (Expected Left On Base)
    # Pitchers with high K% naturally strand more runners; high BB% strands fewer.
    df['LOB%'] = lg_LOB + (df['K%'] - 0.22) * 0.4 - (df['BB%'] - 0.08) * 0.4
    df['LOB%'] = df['LOB%'].clip(0.60, 0.90) # Bound to realistic extremes
    
    # 8. Reconstruct ERA
    # Base ERA off FIP, dynamically adjusted for the pitcher's projected BABIP and LOB%
    babip_penalty = (df['BABIP'] - lg_BABIP) * 2.5
    lob_penalty = (lg_LOB - df['LOB%']) * 2.5
    
    df['ERA'] = df['FIP'] + babip_penalty + lob_penalty
    
    return df