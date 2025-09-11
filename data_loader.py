import streamlit as st
import pandas as pd
import os
import glob

@st.cache_data
def load_data():
    shots = pd.read_csv('data/euro2024_shots.csv')
    matches = pd.read_csv('data/euro2024_matches.csv')
    passes = pd.read_csv('data/euro2024_passes.csv')
    lineups = pd.read_csv('data/lineups.csv')
    groups = pd.read_csv('data/euro2024_groups.csv')
    events = load_all_events()

    # Process shots
    shots_euro = shots[shots['type'] == 'Shot'].copy()
    # Use a copy for merging that doesn't affect the main 'matches' df
    matches_for_shots = matches[['match_id', 'home_team', 'away_team']].copy()
    
    def get_opponent(row):
        if row['team'] == row['home_team']:
            return row['away_team']
        elif row['team'] == row['away_team']:
            return row['home_team']
        return None

    shots_euro = shots_euro.merge(matches_for_shots, on='match_id', how='left')
    shots_euro['opponent_team'] = shots_euro.apply(get_opponent, axis=1)
    
    # Return the original, complete matches dataframe
    return shots_euro, matches, passes, lineups, groups, events

def load_all_events(data_folder='data/'):
    events_file = os.path.join(data_folder, 'euro2024_events.csv')
    if os.path.exists(events_file):
        return pd.read_csv(events_file, low_memory=False)
    return pd.DataFrame()


def calculate_xgchain(events, match_id, team_name):
    # Filter events for the match and team
    match_events = events[events['match_id'] == match_id]
    team_events = match_events[match_events['team'] == team_name]
    # Find all possession chains ending in a shot
    chains = team_events.groupby('possession')
    xgchain = {}
    for possession, chain in chains:
        shot_rows = chain[chain['type'] == 'Shot']
        if not shot_rows.empty:
            xg = shot_rows['shot_statsbomb_xg'].max() if 'shot_statsbomb_xg' in shot_rows and not shot_rows['shot_statsbomb_xg'].isnull().all() else 0
            players_in_chain = chain['player'].dropna().unique()
            for player in players_in_chain:
                xgchain[player] = xgchain.get(player, 0) + xg
    return xgchain
