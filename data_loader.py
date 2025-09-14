import streamlit as st
import pandas as pd
import numpy as np
import os
import glob
import ast

@st.cache_data
def load_data():
    shots = pd.read_csv('data/euro2024_shots.csv')
    matches = pd.read_csv('data/euro2024_matches.csv')
    passes = pd.read_csv('data/euro2024_passes.csv')
    lineups = pd.read_csv('data/lineups.csv')
    groups = pd.read_csv('data/euro2024_groups.csv')
    events = pd.read_csv('data/euro2024_events.csv')
    fifty_fifties = pd.read_csv('data/euro2024_50_50s.csv')

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
    
    # Process 50_50s
    fifty_fifties['50_50_parsed'] = fifty_fifties['50_50'].apply(
        lambda s: ast.literal_eval(s) if pd.notna(s) and isinstance(s, str) else None
    )
    fifty_fifties['outcome'] = fifty_fifties['50_50_parsed'].apply(lambda d: d.get('outcome') if isinstance(d, dict) else None)
    fifty_fifties['outcome_name'] = fifty_fifties['outcome'].apply(lambda o: o.get('name') if isinstance(o, dict) else None)

    # Calculate successful, unsuccessful, and total 50_50s for each player
    fifty_fifty_stats = fifty_fifties.groupby('player')['outcome_name'].value_counts().unstack(fill_value=0)
    fifty_fifty_stats['successful'] = fifty_fifty_stats.get('Won', 0) + fifty_fifty_stats.get('Success To Team', 0)
    fifty_fifty_stats['unsuccessful'] = fifty_fifty_stats.get('Lost', 0) + fifty_fifty_stats.get('Failure To Team', 0)
    fifty_fifty_stats['total'] = fifty_fifty_stats['successful'] + fifty_fifty_stats['unsuccessful']

    # Reset index for easier merging
    fifty_fifty_stats = fifty_fifty_stats[['successful', 'unsuccessful', 'total']].reset_index()

    # Replace competition_stage for group stage matches with their group letter
    group_teams = groups[['team', 'group']].dropna()
    matches['competition_stage'] = matches.apply(
        lambda row: group_teams[group_teams['team'] == row['home_team']]['group'].iloc[0]
        if row['competition_stage'] == 'Group Stage' and not group_teams[group_teams['team'] == row['home_team']].empty
        else row['competition_stage'],
        axis=1
    )

    # Return the original, complete matches dataframe
    return shots_euro, matches, passes, lineups, groups, events, fifty_fifties, fifty_fifty_stats

def calculate_xgchain(events, match_id, team_name):
    # Filter events for the match and team
    match_events = events[events['match_id'] == match_id]
    team_events = match_events[match_events['team'] == team_name]
    # Find all possession chains ending in a shot
    chains = team_events.groupby('possession')
    xgchain = {}
    for possession, chain in chains:
        shot_rows = chain[chain['type'] == 'Shot']
        xg = shot_rows['shot_statsbomb_xg'].max() if 'shot_statsbomb_xg' in shot_rows and not shot_rows['shot_statsbomb_xg'].isnull().all() else 0
        players_in_chain = chain['player'].dropna().unique()
        for player in players_in_chain:
            xgchain[player] = xgchain.get(player, 0) + xg
    return xgchain


def extract_shot_coordinates(shots_df):
    """Extract x, y, z coordinates from shot_end_location"""
    def extract_xyz(val):
        if isinstance(val, (list, tuple)) and len(val) == 3:
            return val
        if isinstance(val, str):
            try:
                arr = eval(val)
                if isinstance(arr, (list, tuple)) and len(arr) >= 3:
                    return arr[:3]
            except Exception:
                pass
        return [np.nan, np.nan, np.nan]

    xyz = shots_df['shot_end_location'].apply(extract_xyz).tolist()
    shots_df[['end_x', 'end_y', 'end_z']] = pd.DataFrame(xyz, index=shots_df.index)
    return shots_df

def calculate_progressive_passes(team_passes):
    """Calculate progressive passes for each player"""
    progressive_passes = {}
    
    for player in team_passes['player'].unique():
        if pd.isna(player):
            continue
            
        player_passes = team_passes[team_passes['player'] == player]
        progressive_count = 0
        
        for _, pass_row in player_passes.iterrows():
            start_loc = eval(pass_row['location']) if isinstance(pass_row['location'], str) else pass_row['location']
            end_loc = eval(pass_row['pass_end_location']) if isinstance(pass_row['pass_end_location'], str) else pass_row['pass_end_location']
                    
            if start_loc and end_loc and len(start_loc) >= 2 and len(end_loc) >= 2:
                    # Progressive if moves ball forward by at least 10 meters towards goal
                forward_progress = end_loc[0] - start_loc[0]
                if forward_progress >= 10:
                    progressive_count += 1

        
        progressive_passes[player] = progressive_count
    
    return progressive_passes

def calculate_pass_success_under_pressure(team_passes):
    """Calculate pass success rate under pressure for each player"""
    pass_success_under_pressure = {}
    
    for player in team_passes['player'].unique():
        if pd.isna(player):
            continue
            
        player_passes = team_passes[team_passes['player'] == player]
        
        # Check multiple possible column names for pressure indication
        pressure_conditions = []
        pressure_conditions.append(player_passes['under_pressure'] == True)

        
        # Combine all pressure conditions with OR
        combined_pressure = pressure_conditions[0]
        for condition in pressure_conditions[1:]:
            combined_pressure = combined_pressure | condition
            
        under_pressure_passes = player_passes[combined_pressure]

        
        if len(under_pressure_passes) > 0:
            # Calculate success rate using corrected logic
            successful_count = calculate_successful_passes(under_pressure_passes)
            total_pressure_passes = len(under_pressure_passes)
            success_rate = (successful_count / total_pressure_passes) * 100
            pass_success_under_pressure[player] = (success_rate, total_pressure_passes)
        else:
            pass_success_under_pressure[player] = (0, 0)
    
    return pass_success_under_pressure

def calculate_successful_passes(passes_df):
    """
    Calculate number of successful passes.
    Passes missing an "outcome" are completed passes.
    Only passes with explicit negative outcomes are considered unsuccessful.
    """
    # Passes with explicit negative outcomes are unsuccessful
    negative_outcomes = ['Incomplete', 'Injury Clearance', 'Out', 'Pass Offside', 'Unknown']
    
    unsuccessful_passes = passes_df[
        passes_df['pass_outcome'].str.lower().isin([outcome.lower() for outcome in negative_outcomes])
    ]
    
    successful_count = len(passes_df) - len(unsuccessful_passes)
    return successful_count

def calculate_pass_accuracy(team_passes):
    """Calculate pass accuracy for each player using corrected logic"""
    pass_accuracy = {}
    
    for player in team_passes['player'].unique():
            
        player_passes = team_passes[team_passes['player'] == player]
        
        if len(player_passes) > 0:
            successful_count = calculate_successful_passes(player_passes)
            accuracy = (successful_count / len(player_passes)) * 100
            pass_accuracy[player] = accuracy
        else:
            pass_accuracy[player] = 0
    
    return pass_accuracy

def calculate_player_positions(team_passes, starting_11):
    """Calculate average positions for players"""
    player_positions = {}
    
    for player in starting_11:
        player_passes = team_passes[team_passes['player'] == player]
        locs = player_passes['location'].apply(lambda val: eval(val) if isinstance(val, str) else val)
        xs = [loc[0] for loc in locs if loc and len(loc) > 1]
        ys = [loc[1] for loc in locs if loc and len(loc) > 1]
        if xs and ys:
            # Flip y-axis for correct orientation
            player_positions[player] = (sum(xs)/len(xs), 80 - (sum(ys)/len(ys)))
        else:
            player_positions[player] = (0, 0)
    
    return player_positions

def calculate_pass_counts(team_passes, starting_11):
    """Calculate pass counts between players"""
    pass_counts = {}
    
    for _, row in team_passes.iterrows():
        passer = row['player']
        recipient = row['pass_recipient']
        if pd.notnull(passer) and pd.notnull(recipient) and passer in starting_11 and recipient in starting_11:
            key = (passer, recipient)
            pass_counts[key] = pass_counts.get(key, 0) + 1
    
    return pass_counts

def get_player_mappings(lineups, match_id, team_name, starting_11):
    """Get player number and nickname mappings"""
    if lineups.empty:
        team_numbers = {p: p for p in starting_11}
        team_nicknames = {p: p.split()[-1] if p else "" for p in starting_11}
    else:
        team_lineup = lineups[(lineups['match_id'] == match_id) & (lineups['team_name'] == team_name)]
        if team_lineup.empty:
            team_numbers = {p: p for p in starting_11}
            team_nicknames = {p: p.split()[-1] if p else "" for p in starting_11}
        else:
            team_numbers = {p['player_name']: str(p['jersey_number']) for _, p in team_lineup.iterrows()}
            team_nicknames = {
                p['player_name']: p['player_nickname'] if pd.notnull(p['player_nickname']) else p['player_name']
                for _, p in team_lineup.iterrows()
            }
    
    return team_numbers, team_nicknames

def calculate_minutes_played(team_passes_or_events, player_name):
    """Calculate minutes played for a player"""
    player_events = team_passes_or_events[team_passes_or_events['player'] == player_name]
    if len(player_events) > 0:
        min_minute = player_events['minute'].min()
        max_minute = player_events['minute'].max()
        minutes_played = max_minute - min_minute + 1  # +1 to include both start and end minutes
        # Cap at 90 minutes for regular time (could be extended for extra time)
        minutes_played = min(minutes_played, 90)
    else:
        minutes_played = 0
    
    return minutes_played

def calculate_quality_scores(team_passes, starting_11, progressive_passes, xgchain_data):
    """Calculate composite quality scores for players"""
    quality_scores = {}
    pass_accuracy = calculate_pass_accuracy(team_passes)
    
    for player in starting_11:
        # Get pass accuracy
        accuracy = pass_accuracy.get(player, 0)
        
        # Get progressive pass rate
        total_passes_player = len(team_passes[team_passes['player'] == player])
        prog_passes = progressive_passes.get(player, 0)
        progressive_rate = (prog_passes / total_passes_player * 100) if total_passes_player > 0 else 0
        
        # Get xGChain value
        xgchain_value = xgchain_data.get(player, 0) if xgchain_data else 0
        
        # Normalize xGChain to 0-100 scale for composite score
        max_xgchain = max(xgchain_data.values()) if xgchain_data else 1
        normalized_xgchain = (xgchain_value / max_xgchain * 100) if max_xgchain > 0 else 0
        
        # Composite quality score (weighted average)
        quality_score = (accuracy * 0.3) + (progressive_rate * 0.4) + (normalized_xgchain * 0.3)
        quality_scores[player] = quality_score
    
    return quality_scores

def prepare_shot_map_data(shots_df, jitter=0.03):
    """Prepare shot data for visualization"""
    shots = shots_df.copy()
    
    if shots.empty:
        return None
    
    # Extract coordinates
    shots = extract_shot_coordinates(shots)
    
    # Add goal indicator and jitter
    shots['is_goal'] = shots['shot_outcome'].astype(str).eq('Goal')
    shots['plot_y'] = shots['end_y'] + np.random.uniform(-jitter, jitter, len(shots))
    shots['plot_z'] = shots['end_z'] + np.random.uniform(-jitter, jitter, len(shots))
    
    # Calculate statistics
    total = len(shots)
    goals = int(shots['is_goal'].sum())
    not_goals = total - goals
    goal_rate = 100 * goals / total if total else 0
    
    miss_types = shots.loc[~shots['is_goal'], 'shot_outcome'].value_counts().sort_values(ascending=False)
    body_part_counts = shots['shot_body_part'].value_counts()
    
    shootout_count = int((shots['period'] == 5).sum())
    regular_count = total - shootout_count
    
    is_penalty = shots['shot_type'].eq('Penalty').all()
    
    return {
        'shots': shots,
        'total': total,
        'goals': goals,
        'not_goals': not_goals,
        'goal_rate': goal_rate,
        'miss_types': miss_types,
        'body_part_counts': body_part_counts,
        'is_penalty': is_penalty,
        'shootout_count': shootout_count,
        'regular_count': regular_count
    }

def prepare_pass_network_data(passes_euro, lineups, match_id, team_name, xgchain_data):
    """Prepare all data needed for pass network visualization"""
    team_passes = passes_euro[(passes_euro['match_id'] == match_id) & (passes_euro['team'] == team_name)]
    
    if team_passes.empty:
        return None
    
    # Get starting 11
    starting_11 = team_passes.sort_values('minute').drop_duplicates('player').head(11)['player'].tolist()
    
    # Calculate all required data
    player_positions = calculate_player_positions(team_passes, starting_11)
    pass_counts = calculate_pass_counts(team_passes, starting_11)
    team_numbers, team_nicknames = get_player_mappings(lineups, match_id, team_name, starting_11)
    progressive_passes = calculate_progressive_passes(team_passes)
    pass_success_under_pressure = calculate_pass_success_under_pressure(team_passes)
    quality_scores = calculate_quality_scores(team_passes, starting_11, progressive_passes, xgchain_data)
    pass_accuracy = calculate_pass_accuracy(team_passes)
    
    return {
        'team_passes': team_passes,
        'starting_11': starting_11,
        'player_positions': player_positions,
        'pass_counts': pass_counts,
        'team_numbers': team_numbers,
        'team_nicknames': team_nicknames,
        'progressive_passes': progressive_passes,
        'pass_success_under_pressure': pass_success_under_pressure,
        'quality_scores': quality_scores,
        'pass_accuracy': pass_accuracy
    }
    


def calculate_tournament_xgchain(events, team_name):
    """Calculate xGChain for a team across the entire tournament"""
    team_events = events[events['team'] == team_name]
    
    # Find all possession chains ending in a shot across all matches
    chains = team_events.groupby(['match_id', 'possession'])
    xgchain = {}
    
    for (match_id, possession), chain in chains:
        shot_rows = chain[chain['type'] == 'Shot']
        xg = shot_rows['shot_statsbomb_xg'].max() if 'shot_statsbomb_xg' in shot_rows and not shot_rows['shot_statsbomb_xg'].isnull().all() else 0
        players_in_chain = chain['player'].dropna().unique()
        for player in players_in_chain:
            xgchain[player] = xgchain.get(player, 0) + xg
    
    return xgchain

def prepare_tournament_pass_network_data(passes_euro, lineups, team_name, xgchain_data):
    """Prepare tournament-wide pass network data for a team"""
    team_passes = passes_euro[passes_euro['team'] == team_name]
    
    if team_passes.empty:
        return None
    
    # Get players who played the most minutes (regular starters)
    player_minutes = {}
    for player in team_passes['player'].dropna().unique():
        minutes = calculate_minutes_played(team_passes, player)
        player_minutes[player] = minutes
    
    # Get top 11 players by minutes played
    starting_11 = sorted(player_minutes.keys(), key=lambda x: player_minutes[x], reverse=True)[:11]
    
    # Calculate all required data
    player_positions = calculate_tournament_player_positions(team_passes, starting_11)
    pass_counts = calculate_pass_counts(team_passes, starting_11)
    team_numbers, team_nicknames = get_tournament_player_mappings(lineups, team_name, starting_11)
    progressive_passes = calculate_progressive_passes(team_passes)
    pass_success_under_pressure = calculate_pass_success_under_pressure(team_passes)
    quality_scores = calculate_quality_scores(team_passes, starting_11, progressive_passes, xgchain_data)
    pass_accuracy = calculate_pass_accuracy(team_passes)
    
    return {
        'team_passes': team_passes,
        'starting_11': starting_11,
        'player_positions': player_positions,
        'pass_counts': pass_counts,
        'team_numbers': team_numbers,
        'team_nicknames': team_nicknames,
        'progressive_passes': progressive_passes,
        'pass_success_under_pressure': pass_success_under_pressure,
        'quality_scores': quality_scores,
        'pass_accuracy': pass_accuracy
    }

def calculate_tournament_player_positions(team_passes, starting_11):
    """Calculate average positions for players across all tournament matches"""
    player_positions = {}
    
    for player in starting_11:
        player_passes = team_passes[team_passes['player'] == player]
        locs = player_passes['location'].apply(lambda val: eval(val) if isinstance(val, str) else val)
        xs = [loc[0] for loc in locs if loc and len(loc) > 1]
        ys = [loc[1] for loc in locs if loc and len(loc) > 1]
        if xs and ys:
            # Flip y-axis for correct orientation
            player_positions[player] = (sum(xs)/len(xs), 80 - (sum(ys)/len(ys)))
        else:
            player_positions[player] = (60, 40)  # Default center position
    
    return player_positions

def get_tournament_player_mappings(lineups, team_name, starting_11):
    """Get player number and nickname mappings across the tournament"""
    if lineups.empty:
        team_numbers = {p: str(i+1) for i, p in enumerate(starting_11)}
        team_nicknames = {p: p.split()[-1] if p else "" for p in starting_11}
    else:
        team_lineup = lineups[lineups['team_name'] == team_name]
        
        # Try to get consistent jersey numbers across matches
        team_numbers = {}
        team_nicknames = {}
        
        for player in starting_11:
            player_lineups = team_lineup[team_lineup['player_name'] == player]
            if not player_lineups.empty:
                # Use the most common jersey number for this player
                jersey_numbers = player_lineups['jersey_number'].value_counts()
                most_common_jersey = jersey_numbers.index[0] if not jersey_numbers.empty else len(team_numbers) + 1
                team_numbers[player] = str(most_common_jersey)
                
                # Get nickname
                nicknames = player_lineups['player_nickname'].dropna()
                nickname = nicknames.iloc[0] if not nicknames.empty else player
                team_nicknames[player] = nickname
            else:
                team_numbers[player] = str(len(team_numbers) + 1)
                team_nicknames[player] = player.split()[-1] if player else ""
    
    return team_numbers, team_nicknames

def calculate_player_radar_stats(all_events, passes_euro, shots_euro, match_id, team_name, player_name):
    """
    Calculate comprehensive player statistics for radar chart
    """
    # Filter data for specific team and player (and match if specified)
    if match_id is not None:
        player_events = all_events[(all_events['match_id'] == match_id) & 
                              (all_events['team'] == team_name) & 
                              (all_events['player'] == player_name)]
        
        player_passes = passes_euro[(passes_euro['match_id'] == match_id) & 
                              (passes_euro['team'] == team_name) & 
                              (passes_euro['player'] == player_name)]
        
        player_shots = shots_euro[(shots_euro['match_id'] == match_id) & 
                            (shots_euro['team'] == team_name) & 
                            (shots_euro['player'] == player_name)]
    else:
        # Tournament-wide stats
        player_events = all_events[(all_events['team'] == team_name) & 
                              (all_events['player'] == player_name)]
        
        player_passes = passes_euro[(passes_euro['team'] == team_name) & 
                              (passes_euro['player'] == player_name)]
        
        player_shots = shots_euro[(shots_euro['team'] == team_name) & 
                            (shots_euro['player'] == player_name)]
    
    if player_events.empty:
        return None
    
    stats = {}
    
    # PASSING METRICS
    total_passes = len(player_passes)
    stats['total_passes'] = total_passes
    
    # Pass accuracy
    if total_passes > 0:
        successful_passes = calculate_successful_passes(player_passes)
        stats['pass_accuracy'] = (successful_passes / total_passes) * 100
    else:
        stats['pass_accuracy'] = 0
    
    # Key passes (passes leading to shots)
    key_passes = player_events[player_events['pass_shot_assist'] == True]
    stats['key_passes'] = len(key_passes)
    
    # Progressive passes
    progressive_passes_dict = calculate_progressive_passes(player_passes)
    stats['progressive_passes'] = progressive_passes_dict.get(player_name, 0)
    
    # CARRIES & DRIBBLING
    carries = player_events[player_events['type'] == 'Carry']
    stats['carries'] = len(carries)
    
    # Progressive carries (carries that move ball forward significantly)
    progressive_carries = 0
    for _, carry in carries.iterrows():
        if pd.notna(carry.get('carry_end_location')) and pd.notna(carry.get('location')):
            try:
                start_loc = eval(carry['location']) if isinstance(carry['location'], str) else carry['location']
                end_loc = eval(carry['carry_end_location']) if isinstance(carry['carry_end_location'], str) else carry['carry_end_location']
                
                if start_loc and end_loc and len(start_loc) >= 2 and len(end_loc) >= 2:
                    forward_progress = end_loc[0] - start_loc[0]
                    if forward_progress >= 15:  # Progressive carry if 15+ meters forward
                        progressive_carries += 1
            except:
                continue
    
    stats['progressive_carries'] = progressive_carries
    
    # Carries into final third
    final_third_carries = 0
    for _, carry in carries.iterrows():
        if pd.notna(carry.get('carry_end_location')):
            try:
                end_loc = eval(carry['carry_end_location']) if isinstance(carry['carry_end_location'], str) else carry['carry_end_location']
                if end_loc and len(end_loc) >= 2 and end_loc[0] >= 80:  # Final third starts at x=80
                    final_third_carries += 1
            except:
                continue
    
    stats['final_third_carries'] = final_third_carries
    
    # Carries into penalty area
    penalty_area_carries = 0
    for _, carry in carries.iterrows():
        if pd.notna(carry.get('carry_end_location')):
            try:
                end_loc = eval(carry['carry_end_location']) if isinstance(carry['carry_end_location'], str) else carry['carry_end_location']
                if end_loc and len(end_loc) >= 2 and end_loc[0] >= 102 and 18 <= end_loc[1] <= 62:  # Penalty area
                    penalty_area_carries += 1
            except:
                continue
    
    stats['penalty_area_carries'] = penalty_area_carries
    
    dribbles = player_events[player_events['type'] == 'Dribble']
    successful_dribbles = dribbles[dribbles['dribble_outcome'].isin(['Complete', 'Successful'])]
    stats['dribbles_completed'] = len(successful_dribbles)
    stats['dribble_success_rate'] = (len(successful_dribbles) / len(dribbles) * 100) if len(dribbles) > 0 else 0
    
    # SHOOTING & ATTACKING
    stats['shots'] = len(player_shots)
    stats['goals'] = len(player_shots[player_shots['shot_outcome'] == 'Goal'])
    # Non-penalty goals
    non_penalty_goals = player_shots[(player_shots['shot_outcome'] == 'Goal') & (player_shots['shot_type'] != 'Penalty')]
    stats['non_penalty_goals'] = len(non_penalty_goals)
    # Non-penalty xG (exclude penalty shots from xG calculation)
    non_penalty_shots = player_shots[player_shots['shot_type'] != 'Penalty']
    stats['xg'] = non_penalty_shots['shot_statsbomb_xg'].sum() if 'shot_statsbomb_xg' in non_penalty_shots.columns else 0
    stats['non_penalty_xg'] = stats['xg']  # Since we already excluded penalties above

    # Assists
    assists = player_events[player_events['pass_goal_assist'] == True]
    stats['assists'] = len(assists)
    
    # Expected assists (xA) - sum of xG from shots that came from this player's passes
    expected_assists = 0
    key_passes_with_xg = player_events[player_events['pass_shot_assist'] == True]
    if not key_passes_with_xg.empty:
        # For each key pass, try to find the corresponding shot and get its xG
        for _, key_pass in key_passes_with_xg.iterrows():
            # Look for shots shortly after this key pass in the same possession
            match_events = all_events[all_events['match_id'] == key_pass['match_id']]
            possession_shots = match_events[
                (match_events['possession'] == key_pass['possession']) &
                (match_events['type'] == 'Shot') &
                (match_events['minute'] >= key_pass['minute'])
            ]
            if not possession_shots.empty:
                # Add the xG of the first shot after this key pass
                shot_xg = possession_shots.iloc[0].get('shot_statsbomb_xg', 0)
                if pd.notna(shot_xg):
                    expected_assists += shot_xg
    
    stats['expected_assists'] = expected_assists
    
    # Non-penalty G+A and xG+xA
    stats['non_penalty_g_a'] = stats['non_penalty_goals'] + stats['assists']
    stats['non_penalty_xg_xa'] = stats['non_penalty_xg'] + stats['expected_assists']

    # 50/50s WON
    fifty_fifties = player_events[player_events['type'] == 'Duel']
    fifty_fifties_won = fifty_fifties[fifty_fifties['duel_outcome'].isin(['Won', 'Success To Team'])]
    stats['50_50s_won'] = len(fifty_fifties_won)
    
    # PRESSURE PLAY
    pressure_events = player_events[player_events['type'] == 'Pressure']
    stats['pressures'] = len(pressure_events)
    
    # Pass success under pressure
    pressure_stats = calculate_pass_success_under_pressure(player_passes)
    stats['pass_success_under_pressure'] = pressure_stats.get(player_name, (0, 0))[0]
    
    # DEFENSIVE ACTIONS
    tackles = player_events[
        (player_events['type'] == 'Duel') &
        (player_events['duel_type'] == 'Tackle') &
        (~player_events['duel_outcome'].isin(['Lost In Play']))
    ]
    stats['tackles'] = len(tackles)
    
    interceptions = player_events[player_events['type'] == 'Interception']
    stats['interceptions'] = len(interceptions)
    
    blocks = player_events[player_events['type'] == 'Block']
    stats['blocks'] = len(blocks)
    
    recoveries = player_events[player_events['type'] == 'Ball Recovery']
    stats['recoveries'] = len(recoveries)
    
    clearances = player_events[player_events['type'] == 'Clearance']
    stats['clearances'] = len(clearances)
    
    # Dribbled past (defensive)
    dribbled_past = player_events[player_events['type'] == 'Dribbled Past']
    stats['dribbled_past'] = len(dribbled_past)
    
    # CREATIVITY METRICS
    # Through balls and long passes
    through_balls = player_passes[player_passes['pass_type'].str.contains('Through Ball', na=False)]
    stats['through_balls'] = len(through_balls)
    
    # Long passes (over 30 meters)
    long_passes = player_passes.copy()
    long_pass_count = 0
    for _, pass_row in long_passes.iterrows():
        start_loc = eval(pass_row['location']) if isinstance(pass_row['location'], str) else pass_row['location']
        end_loc = eval(pass_row['pass_end_location']) if isinstance(pass_row['pass_end_location'], str) else pass_row['pass_end_location']
        
        if start_loc and end_loc and len(start_loc) >= 2 and len(end_loc) >= 2:
            distance = ((end_loc[0] - start_loc[0])**2 + (end_loc[1] - start_loc[1])**2)**0.5
            if distance >= 30:
                long_pass_count += 1
    
    stats['long_passes'] = long_pass_count
    
    # Passes into final third
    final_third_passes = 0
    for _, pass_row in player_passes.iterrows():
        end_loc = eval(pass_row['pass_end_location']) if isinstance(pass_row['pass_end_location'], str) else pass_row['pass_end_location']
        if end_loc and len(end_loc) >= 2 and end_loc[0] >= 80:  # Final third starts at x=80
            final_third_passes += 1
    
    stats['final_third_passes'] = final_third_passes
    
    # Calculate total minutes played across all matches
    if match_id is not None:
        minutes_played = calculate_minutes_played(player_events, player_name)
    else:
        # For tournament-wide, sum minutes across all matches
        total_minutes = 0
        for match in player_events['match_id'].unique():
            match_events = player_events[player_events['match_id'] == match]
            match_minutes = calculate_minutes_played(match_events, player_name)
            total_minutes += match_minutes
        minutes_played = total_minutes
    
    stats['minutes_played'] = minutes_played
    
    # Convert to per 90 minute rates for ALL stats including carry metrics
    if minutes_played > 0:
        rate_multiplier = 90 / minutes_played
        
        # Passing stats per 90
        stats['passes_per_90'] = total_passes * rate_multiplier
        stats['key_passes_per_90'] = stats['key_passes'] * rate_multiplier
        stats['progressive_passes_per_90'] = stats['progressive_passes'] * rate_multiplier
        
        # Attacking stats per 90
        stats['shots_per_90'] = stats['shots'] * rate_multiplier
        stats['goals_per_90'] = stats['goals'] * rate_multiplier
        stats['non_penalty_goals_per_90'] = stats['non_penalty_goals'] * rate_multiplier
        stats['assists_per_90'] = stats['assists'] * rate_multiplier
        stats['xg_per_90'] = stats['xg'] * rate_multiplier
        stats['non_penalty_xg_per_90'] = stats['non_penalty_xg'] * rate_multiplier
        
        # Dribbling and carry stats per 90
        stats['dribbles_completed_per_90'] = stats['dribbles_completed'] * rate_multiplier
        stats['carries_per_90'] = stats['carries'] * rate_multiplier
        stats['progressive_carries_per_90'] = stats['progressive_carries'] * rate_multiplier
        stats['final_third_carries_per_90'] = stats['final_third_carries'] * rate_multiplier
        stats['penalty_area_carries_per_90'] = stats['penalty_area_carries'] * rate_multiplier
        
        # Defensive stats per 90
        stats['tackles_per_90'] = stats['tackles'] * rate_multiplier
        stats['interceptions_per_90'] = stats['interceptions'] * rate_multiplier
        stats['recoveries_per_90'] = stats['recoveries'] * rate_multiplier
        stats['blocks_per_90'] = stats['blocks'] * rate_multiplier
        stats['clearances_per_90'] = stats['clearances'] * rate_multiplier
        stats['50_50s_won_per_90'] = stats['50_50s_won'] * rate_multiplier
        stats['pressures_per_90'] = stats['pressures'] * rate_multiplier
        
        # Creativity stats per 90
        stats['through_balls_per_90'] = stats['through_balls'] * rate_multiplier
        stats['long_passes_per_90'] = stats['long_passes'] * rate_multiplier
        stats['final_third_passes_per_90'] = stats['final_third_passes'] * rate_multiplier
        
        # Add per-90 rates for new metrics
        stats['expected_assists_per_90'] = stats['expected_assists'] * rate_multiplier
        stats['non_penalty_xg_xa_per_90'] = stats['non_penalty_xg_xa'] * rate_multiplier
        
    else:
        # Set all per-90 stats to 0 if no minutes played (now includes carry stats)
        for stat in ['passes', 'key_passes', 'progressive_passes', 'shots', 'goals', 'non_penalty_goals', 'assists', 'xg', 'non_penalty_xg',
                    'dribbles_completed', 'carries', 'progressive_carries', 'final_third_carries', 'penalty_area_carries', 
                    'tackles', 'interceptions', 'recoveries', 'blocks', 'clearances', '50_50s_won', 'pressures',
                    'through_balls', 'long_passes', 'final_third_passes']:
            stats[f'{stat}_per_90'] = 0
    
    return stats

def normalize_radar_stats(all_player_stats):
    """
    Normalize player stats to 0-100 scale for radar chart using percentile ranks
    Creates aggregated categories for radar visualization
    """
    if not all_player_stats:
        return {}
    
    # Use percentile-based normalization instead of min-max
    import scipy.stats as stats
    
    # Calculate percentile ranks for each stat using per-90 stats (now includes carry stats)
    stats_to_track = [
        'tackles_per_90', 'interceptions_per_90', 'recoveries_per_90', 'blocks_per_90', 'clearances_per_90',
        'passes_per_90', 'pass_accuracy', 'progressive_passes_per_90', 'key_passes_per_90',
        'non_penalty_goals_per_90', 'assists_per_90', 'xg_per_90',
        'dribbles_completed_per_90', 'dribble_success_rate', 'carries_per_90', 'progressive_carries_per_90', 
        'final_third_carries_per_90', 'penalty_area_carries_per_90',
        '50_50s_won_per_90', 'pressures_per_90',
        'through_balls_per_90', 'long_passes_per_90', 'final_third_passes_per_90'
    ]
    
    # Create arrays of values for percentile calculation
    stat_arrays = {}
    for stat in stats_to_track:
        values = [player_stats.get(stat, 0) for player_stats in all_player_stats.values()]
        stat_arrays[stat] = np.array(values)
    
    # Normalize each player's stats and create aggregated categories
    normalized_stats = {}
    for player_name, player_stats in all_player_stats.items():
        normalized = {}
        
        # Helper function to get percentile rank (0-100)
        def get_percentile_rank(value, stat_name):
            if stat_name not in stat_arrays:
                return 0
            values_array = stat_arrays[stat_name]
            if len(values_array) <= 1:
                return 50
            percentile = stats.percentileofscore(values_array, value, kind='rank')
            return percentile
        
        # 1. Defending (combines all defensive actions)
        defending_components = []
        for stat in ['tackles_per_90', 'interceptions_per_90', 'pressures_per_90', '50_50s_won_per_90', 
                    'blocks_per_90', 'clearances_per_90', 'recoveries_per_90']:
            percentile = get_percentile_rank(player_stats.get(stat, 0), stat)
            defending_components.append(percentile)
        normalized['defending'] = sum(defending_components) / len(defending_components) if defending_components else 0
        
        # 2. Passing (general passing including progressive, key passes, and accuracy)
        progressive_percentile = get_percentile_rank(player_stats.get('progressive_passes_per_90', 0), 'progressive_passes_per_90')
        key_passes_percentile = get_percentile_rank(player_stats.get('key_passes_per_90', 0), 'key_passes_per_90')
        pass_accuracy = player_stats.get('pass_accuracy', 0)
        pass_accuracy_percentile = get_percentile_rank(pass_accuracy, 'pass_accuracy')
        passes_volume_percentile = get_percentile_rank(player_stats.get('passes_per_90', 0), 'passes_per_90')
        
        # Weighted average: progressive (30%) + key passes (25%) + accuracy (25%) + volume (20%)
        normalized['passing'] = (progressive_percentile * 0.3 + key_passes_percentile * 0.25 + 
                               pass_accuracy_percentile * 0.25 + passes_volume_percentile * 0.2)
        
        # 3. Ball Carrying (enhanced with progressive carries and attacking carries)
        carries_volume_percentile = get_percentile_rank(player_stats.get('carries_per_90', 0), 'carries_per_90')
        progressive_carries_percentile = get_percentile_rank(player_stats.get('progressive_carries_per_90', 0), 'progressive_carries_per_90')
        final_third_carries_percentile = get_percentile_rank(player_stats.get('final_third_carries_per_90', 0), 'final_third_carries_per_90')
        penalty_area_carries_percentile = get_percentile_rank(player_stats.get('penalty_area_carries_per_90', 0), 'penalty_area_carries_per_90')
        
        # Weight progressive and attacking carries more heavily
        normalized['ball_carrying'] = (carries_volume_percentile * 0.25 + progressive_carries_percentile * 0.35 + 
                                     final_third_carries_percentile * 0.25 + penalty_area_carries_percentile * 0.15)
        
        # 4. Dribbling (focuses purely on dribbling ability)
        dribbles_percentile = get_percentile_rank(player_stats.get('dribbles_completed_per_90', 0), 'dribbles_completed_per_90')
        dribble_success_percentile = get_percentile_rank(player_stats.get('dribble_success_rate', 0), 'dribble_success_rate')
        
        # If player attempted dribbles, weight volume and success rate
        if player_stats.get('dribbles_completed', 0) > 0:
            normalized['dribbling'] = (dribbles_percentile * 0.6 + dribble_success_percentile * 0.4)
        else:
            # If no dribbles attempted, give very low score
            normalized['dribbling'] = dribble_success_percentile * 0.2
        
        # 5. Creativity (through balls, long passes, final third passes)
        through_balls_percentile = get_percentile_rank(player_stats.get('through_balls_per_90', 0), 'through_balls_per_90')
        long_passes_percentile = get_percentile_rank(player_stats.get('long_passes_per_90', 0), 'long_passes_per_90')
        final_third_percentile = get_percentile_rank(player_stats.get('final_third_passes_per_90', 0), 'final_third_passes_per_90')
        
        # Weight different creativity aspects
        normalized['creativity'] = (through_balls_percentile * 0.4 + long_passes_percentile * 0.3 + final_third_percentile * 0.3)
        
        # 6. Work Rate (activity level and energy - now includes carries)
        work_rate_components = []
        # Pressures and 50/50s show active engagement
        for stat in ['pressures_per_90', '50_50s_won_per_90']:
            percentile = get_percentile_rank(player_stats.get(stat, 0), stat)
            work_rate_components.append(percentile)
        
        # Add carries and passes as indicators of involvement
        carries_involvement = get_percentile_rank(player_stats.get('carries_per_90', 0), 'carries_per_90')
        passes_involvement = get_percentile_rank(player_stats.get('passes_per_90', 0), 'passes_per_90')
        work_rate_components.extend([carries_involvement, passes_involvement])
        
        normalized['work_rate'] = sum(work_rate_components) / len(work_rate_components) if work_rate_components else 0
        
        # 7. Goal Threat (non-penalty goals + assists + xG)
        non_penalty_goals_percentile = get_percentile_rank(player_stats.get('non_penalty_goals_per_90', 0), 'non_penalty_goals_per_90')
        assists_percentile = get_percentile_rank(player_stats.get('assists_per_90', 0), 'assists_per_90')
        xg_percentile = get_percentile_rank(player_stats.get('xg_per_90', 0), 'xg_per_90')
        
        # Weight actual output more than expected
        normalized['goal_threat'] = (non_penalty_goals_percentile * 0.4 + assists_percentile * 0.4 + xg_percentile * 0.2)
        
        normalized_stats[player_name] = normalized
    
    return normalized_stats

def normalize_player_data(player_id, tournament_data):
    """
    Collects and normalizes data for a player based on tournament-wide statistics.

    Args:
        player_id (int): The ID of the player to normalize data for.
        tournament_data (pd.DataFrame): DataFrame containing tournament-wide player statistics.

    Returns:
        dict: A dictionary containing normalized values for the player.
    """
    # Define the categories and their corresponding columns
    categories = {
        "defensive_actions": ["tackles", "interceptions", "recoveries", "pressures", "blocks"],
        "passing": ["passes_completed", "passes_attempted", "progressive_passes", "key_passes"],
        "g_a": ["goals", "assists"],
        "dribbling": ["dribbles_attempted", "dribbles_completed"],
        "key_passes": ["key_passes"]
    }

    # Initialize a dictionary to store normalized values
    normalized_data = {}

    # Normalize each category
    for category, columns in categories.items():
        # Aggregate columns within the category
        if category == "passing":
            # Average passing-related stats
            player_value = tournament_data.loc[tournament_data["player_id"] == player_id, columns].mean(axis=1).values[0]
            max_value = tournament_data[columns].mean(axis=1).max()
        else:
            # Sum or use single column for other categories
            player_value = tournament_data.loc[tournament_data["player_id"] == player_id, columns].sum(axis=1).values[0]
            max_value = tournament_data[columns].sum(axis=1).max()

        # Normalize the value
        normalized_data[category] = player_value / max_value if max_value > 0 else 0

    return normalized_data

def get_player_display_name(player_name, lineups_df=None):
    """
    Get player display name, preferring nickname over full name
    """
    if lineups_df is None or lineups_df.empty:
        return player_name
    
    # Find player in lineups
    player_lineups = lineups_df[lineups_df['player_name'] == player_name]
    if not player_lineups.empty:
        # Get nickname if available
        nicknames = player_lineups['player_nickname'].dropna()
        if not nicknames.empty:
            return nicknames.iloc[0]
    
    return player_name

def get_team_player_display_names(team_name, lineups_df):
    """
    Get display names for all players in a team, preferring nicknames
    """
    if lineups_df.empty:
        return {}
    
    team_lineups = lineups_df[lineups_df['team_name'] == team_name]
    display_names = {}
    
    for _, player_row in team_lineups.iterrows():
        player_name = player_row['player_name']
        nickname = player_row['player_nickname']
        
        if pd.notna(nickname):
            display_names[player_name] = nickname
        else:
            display_names[player_name] = player_name
    
    return display_names