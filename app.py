import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt  # Ensure plt is imported for colormap functionality
from data_loader import (
    load_data, 
    calculate_xgchain, 
    prepare_shot_map_data, 
    prepare_pass_network_data,
    calculate_player_radar_stats,
    normalize_radar_stats,
    calculate_tournament_xgchain,
    prepare_tournament_pass_network_data,
    get_player_display_name,
    get_team_player_display_names
)
from visualizations import plotly_penalty_map_center_only, plot_pass_network_plotly, plot_player_radar_chart

# --- Data Loading ---
st.set_page_config(layout="wide")

shots_euro, euro_matches, passes_euro, lineups, euro_groups, all_events, fifty_fifties, fifty_fifty_stats = load_data()

# --- Streamlit UI ---
with st.sidebar:
    st.markdown("# Euro 2024 Analysis")
    
    view_option = st.radio("Select View", ["Shot Map", "Pass Network", "Player Analysis"])
    
    theme = st.selectbox("Theme", ["green", "white", "black"])

    st.markdown("---")

    # Initialize variables to avoid NameError
    selected_team = None
    selected_player = None
    selected_match = None
    match_id = None
    min_passes = 6
    # Initialize comparison variables
    compare_players = False
    selected_team_2 = None
    selected_player_2 = None

    if view_option == "Shot Map":
        st.markdown("## Shot Filters")
        play_patterns = shots_euro['play_pattern'].dropna()
        play_pattern_counts = play_patterns.value_counts().sort_values(ascending=False)
        play_pattern_filtered = play_pattern_counts[play_pattern_counts.index != 'Other']
        play_pattern_options = ['All'] + play_pattern_filtered.index.tolist() + ['Penalty']
        selected_pattern = st.selectbox("Situation", play_pattern_options)

        body_part_counts_sidebar = shots_euro['shot_body_part'].value_counts().sort_values(ascending=False)
        body_part_options = ['All'] + body_part_counts_sidebar.index.tolist()
        selected_body_part = st.selectbox("Body Part", body_part_options, index=0)

        miss_type_counts_sidebar = shots_euro['shot_outcome'].value_counts().sort_values(ascending=False)
        miss_type_options = ['All'] + miss_type_counts_sidebar.index.tolist()
        selected_miss_type = st.selectbox("Outcome", miss_type_options, index=0)

    elif view_option == "Pass Network":
        st.markdown("## Pass Network Filters")
        
        # Define the main tournament phases
        main_phases = [
            "Final", "Semi-finals", "Quarter-finals", "Round of 16", "Group Stage"
        ]
        selected_main_phase = st.selectbox("Select Tournament Phase", main_phases)
        
        if selected_main_phase == "Group Stage":
            # If "Group Stage" is selected, show another dropdown for groups
            group_options = ["Group A", "Group B", "Group C", "Group D", "Group E", "Group F"]
            selected_group = st.selectbox("Select Group", group_options)
            phase_matches = euro_matches[euro_matches['competition_stage'] == selected_group]
        else:
            # Filter matches by the selected main phase
            phase_matches = euro_matches[euro_matches['competition_stage'] == selected_main_phase]
        
        # Filter matches within the selected phase
        match_options = phase_matches.apply(
            lambda row: f"{row['home_team']} vs {row['away_team']}", axis=1
        ).unique()
        selected_match = st.selectbox("Select Match", sorted(match_options))
        
        # Get the match ID for the selected match
        match_row = phase_matches[phase_matches.apply(
            lambda row: f"{row['home_team']} vs {row['away_team']}", axis=1
        ) == selected_match].iloc[0]
        match_id = match_row['match_id']

        # Get teams for the selected match
        selected_team = st.selectbox("Select Team", [match_row['home_team'], match_row['away_team']])
        
        # Calculate max passes for the slider
        max_passes_value = 30  # Default max
        if selected_team:
            team_passes_for_slider = passes_euro[(passes_euro['match_id'] == match_id) & (passes_euro['team'] == selected_team)]
            if not team_passes_for_slider.empty:
                pass_counts = team_passes_for_slider.groupby(['player', 'pass_recipient']).size()
                if not pass_counts.empty:
                    max_passes_value = int(pass_counts.max())

        min_passes = st.slider("Minimum number of passes", 1, max_passes_value, 6)

    elif view_option == "Player Analysis":
        st.markdown("## Player Analysis Filters")
        
        # Team selection
        all_teams = sorted(all_events['team'].dropna().unique())
        selected_team = st.selectbox("Select Team", all_teams)
        
        # Player selection based on selected team
        if selected_team:
            team_players = sorted(all_events[all_events['team'] == selected_team]['player'].dropna().unique())
            
            # Get display names for players (nicknames preferred)
            team_display_names = get_team_player_display_names(selected_team, lineups)
            
            # Create options with display names but keep original names as values
            player_options = [team_display_names.get(player, player) for player in team_players]
            player_name_mapping = {team_display_names.get(player, player): player for player in team_players}
            
            selected_player_display = st.selectbox("Select Player", player_options)
            selected_player = player_name_mapping.get(selected_player_display, selected_player_display)
            
            # Add comparison option
            compare_players = st.checkbox("Compare with another player")
            
            if compare_players:
                # Allow selection from all teams for comparison
                all_teams_2 = sorted(all_events['team'].dropna().unique())
                selected_team_2 = st.selectbox("Select Second Team", all_teams_2, key="team_2")
                
                if selected_team_2:
                    team_players_2 = sorted(all_events[all_events['team'] == selected_team_2]['player'].dropna().unique())
                    
                    # Get display names for second team
                    team_display_names_2 = get_team_player_display_names(selected_team_2, lineups)
                    player_options_2 = [team_display_names_2.get(player, player) for player in team_players_2]
                    player_name_mapping_2 = {team_display_names_2.get(player, player): player for player in team_players_2}
                    
                    selected_player_2_display = st.selectbox("Select Second Player", player_options_2, key="player_2")
                    selected_player_2 = player_name_mapping_2.get(selected_player_2_display, selected_player_2_display)
        else:
            selected_player = None

# --- Main Panel ---
st.title(view_option)

# Assign palette and colors based on theme
if theme == "green":
    palette = "neutral"
    GOAL_COLOR = "#222222"
    MISS_COLOR = "#bbbbbb"
elif theme == "black":
    palette = "classic"
    GOAL_COLOR = "#2ca02c"
    MISS_COLOR = "#d62728"
else: # white
    palette = "vibrant"
    GOAL_COLOR = "#1f77b4"
    MISS_COLOR = "#ff7f0e"

if view_option == "Shot Map":
    # Main panel filter by Country/Player
    country_counts = shots_euro['team'].value_counts().sort_values(ascending=False)
    country_list = ['All'] + country_counts.index.tolist()
    
    # Get all unique players and create display name mapping
    all_players = shots_euro['player'].dropna().unique()
    player_display_mapping = {}
    for player in all_players:
        display_name = get_player_display_name(player, lineups)
        player_display_mapping[display_name] = player
    
    player_display_names = sorted(player_display_mapping.keys())
    player_list = ['All'] + player_display_names
    
    option = st.radio("Filter by:", ["Country", "Player"])
    if option == "Country":
        selected_country = st.selectbox("Select Country", country_list, index=0)
        selected_player = None
    else:
        selected_player_display = st.selectbox("Select Player", player_list, index=0)
        if selected_player_display == 'All':
            selected_player = 'All'
        else:
            selected_player = player_display_mapping.get(selected_player_display, selected_player_display)
        selected_country = None

    # Filtering logic
    filtered_shots = shots_euro.copy()
    if selected_pattern != 'All':
        if selected_pattern == 'Penalty':
            filtered_shots = filtered_shots[filtered_shots['shot_type'] == 'Penalty']
        else:
            filtered_shots = filtered_shots[filtered_shots['play_pattern'] == selected_pattern]
    if selected_body_part != 'All':
        filtered_shots = filtered_shots[filtered_shots['shot_body_part'] == selected_body_part]
    if selected_miss_type != 'All':
        filtered_shots = filtered_shots[filtered_shots['shot_outcome'] == selected_miss_type]
    if option == "Country" and selected_country and selected_country != 'All':
        filtered_shots = filtered_shots[filtered_shots['team'] == selected_country]
    elif option == "Player" and selected_player and selected_player != 'All':
        filtered_shots = filtered_shots[filtered_shots['player'] == selected_player]

    if selected_pattern == 'All':
        shot_type_label = 'Shots'
        title_prefix = 'All shot types'
    elif selected_pattern == 'Penalty':
        shot_type_label = 'Penalties'
        title_prefix = 'All penalties'
    else:
        shot_type_label = selected_pattern
        title_prefix = f'All {selected_pattern} shots'

    title = title_prefix
    fig_data = plotly_penalty_map_center_only(
        filtered_shots, 
        theme=theme, 
        palette=palette, 
        title=title, 
        shot_type_label=shot_type_label,
        GOAL_COLOR=GOAL_COLOR,
        MISS_COLOR=MISS_COLOR
    )
    if fig_data:
        fig, total, goals, not_goals, goal_rate, miss_types, body_part_counts, is_penalty, shootout_count, regular_count = fig_data
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        # Compact KPIs and summaries in columns
        counts_col, body_col, miss_col, pattern_col = st.columns([1.2,1.2,1.2,1.2])
        with counts_col:
            st.markdown(f"### Counts")
            st.markdown(f"**{shot_type_label}: {total}**")
            st.markdown(f"**Goals: {goals} ({goal_rate:.1f}%)**")
            st.markdown(f"**No Goals: {not_goals} ({100 - goal_rate:.1f}%)**")
        with body_col:
            st.markdown("### Body Part")
            for part, count in body_part_counts.items():
                st.markdown(f"{part}: {count}")
        with pattern_col:
            st.markdown(f"### Situation")
            play_pattern_counts = filtered_shots['play_pattern'].value_counts().sort_values(ascending=False)
            for pattern, count in play_pattern_counts.items():
                if pattern == 'Other':
                    penalty_count = filtered_shots[filtered_shots['shot_type'] == 'Penalty'].shape[0]
                    if penalty_count > 0:
                        st.markdown(f"Penalties: {penalty_count}")
                else:
                    st.markdown(f"{pattern}: {count}")
            if is_penalty:
                st.markdown(f"### Type of Play")
                st.markdown(f"Shootout: {shootout_count}")
                st.markdown(f"Regular: {regular_count}")
        with miss_col:  
            st.markdown(f"### Miss Types")
            for miss_type, count in miss_types.items():
                label = "Off Target" if miss_type == "Off T" else miss_type
                st.markdown(f"{label}: {count}")

elif view_option == "Pass Network":
    # Filters remain in the sidebar only
    if selected_team:
        col1, col2, col3 = st.columns([1.5, 4, 1])  # Adjust column widths

        with col1:
            st.markdown("### Key Player Stats")
            xgchain_data = calculate_xgchain(all_events, match_id, selected_team)
            
            # Get starting 11 to filter xgchain data for display
            network_data_for_starters = prepare_pass_network_data(passes_euro, lineups, match_id, selected_team, xgchain_data)
            starting_11 = network_data_for_starters['starting_11'] if network_data_for_starters else []
            
            fig, stats = plot_pass_network_plotly(
                passes_euro=passes_euro,
                lineups=lineups,
                euro_matches=euro_matches,
                match_id=match_id,
                team_name=selected_team,
                min_passes=min_passes,
                theme=theme,
                xgchain_data=xgchain_data,
                node_color=GOAL_COLOR,
                edge_color=MISS_COLOR
            )
            if stats:
                if 'most_passes' in stats:
                    player, value, jersey = stats['most_passes']
                    st.metric(label="Most Passes Given", value=f"#{jersey} {player}", delta=f"{int(value)} passes")
                if 'most_received' in stats:
                    player, value, jersey = stats['most_received']
                    st.metric(label="Most Passes Received", value=f"#{jersey} {player}", delta=f"{int(value)} passes")   
                if 'most_accurate' in stats:
                    player, value, jersey = stats['most_accurate']
                    st.metric(label="Highest Pass Accuracy (>30 attempts)", value=f"#{jersey} {player}", delta=f"{value:.1f}%")                                
                if 'most_key_passes' in stats:
                    player, value, jersey = stats['most_key_passes']
                    st.metric(label="Most Key Passes", value=f"#{jersey} {player}", delta=f"{int(value)} passes")              
                if 'max_centrality' in stats:
                    player, rank, jersey = stats['max_centrality']
                    st.metric(label="Highest Centrality", value=f"#{jersey} {player}", delta=f"Rank {rank}")
                if 'max_betweenness' in stats:
                    player, rank, jersey = stats['max_betweenness']
                    st.metric(label="Best Playmaker", value=f"#{jersey} {player}", delta=f"Rank {rank}")
                if 'most_progressive' in stats:
                    player, value, jersey = stats['most_progressive']
                    st.metric(label="Most Progressive Passes", value=f"#{jersey} {player}", delta=f"{int(value)} passes")
                if 'best_under_pressure' in stats:
                    player, rate, attempts, jersey = stats['best_under_pressure']
                    st.metric(label="Best Under Pressure (>5 attempts)", value=f"#{jersey} {player}", delta=f"{rate:.1f}% ({attempts} att)")
                if 'network_density' in stats:
                    density = stats['network_density']
                    st.metric(label="Network Density", value=f"{density:.1f}%", delta="Team Connectivity")
            else:
                st.info("Not enough data to calculate player stats.")

        with col2:
            st.markdown(f"### {selected_team} — Pass Network")
            if fig:
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        with col3:
            st.markdown("### xGChain")
            if xgchain_data and starting_11:
                # Filter xgchain data to only include starting 11 players
                xgchain_starting_11 = {p: v for p, v in xgchain_data.items() if p in starting_11}

                if xgchain_starting_11:
                    max_xg = max(xgchain_starting_11.values())
                    min_xg = min(xgchain_starting_11.values())

                    # Find players with the highest and lowest xGChain from starting 11
                    max_player = max(xgchain_starting_11, key=xgchain_starting_11.get)
                    min_player = min(xgchain_starting_11, key=xgchain_starting_11.get)

                    # Use nicknames if available
                    team_lineup_info = lineups[(lineups['match_id'] == match_id) & (lineups['team_name'] == selected_team)]

                    def get_player_info(player_name, team_lineup):
                        player_row = team_lineup[team_lineup['player_name'] == player_name]
                        if not player_row.empty:
                            nickname = player_row['player_nickname'].iloc[0] if pd.notna(player_row['player_nickname'].iloc[0]) else player_name
                            jersey = player_row['jersey_number'].iloc[0]
                            return nickname, jersey
                        return player_name, "N/A"

                    max_player_nickname, max_player_jersey = get_player_info(max_player, team_lineup_info)
                    min_player_nickname, min_player_jersey = get_player_info(min_player, team_lineup_info)

                    colormap = plt.cm.get_cmap('viridis', 256)
                    gradient = [colormap(i / 255) for i in range(256)]
                    gradient_colors = [f"rgba({int(c[0]*255)}, {int(c[1]*255)}, {int(c[2]*255)}, 1)" for c in gradient]

                    # Display colormap as a vertical gradient (inverted)
                    st.markdown(f"<div style='text-align:left;'></b> #{max_player_jersey} {max_player_nickname}<br><b>Max:</b> {max_xg:.2f}</div>", unsafe_allow_html=True)
                    st.markdown(
                        f"<div style='height:480px; width:50px; background: linear-gradient(to top, {', '.join(gradient_colors)}); margin-top: 10px; margin-bottom: 10px;'></div>",
                        unsafe_allow_html=True
                    )
                    st.markdown(f"<div style='text-align:left;'></b> #{min_player_jersey} {min_player_nickname}<br><b>Min:</b> {min_xg:.2f}</div>", unsafe_allow_html=True)
    else:
        st.info("Please select a team from the sidebar to view the pass network.")

elif view_option == "Player Analysis":
    if selected_team and selected_player:
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            # Calculate tournament-wide player radar stats for first player
            player_stats = calculate_player_radar_stats(
                all_events, passes_euro, shots_euro, 
                None, selected_team, selected_player
            )
            
            if player_stats:
                # Get all players from both teams for normalization if comparing
                if compare_players and selected_player_2:
                    # Get players from both teams for proper normalization
                    all_team_players = list(all_events[all_events['team'] == selected_team]['player'].dropna().unique())
                    if selected_team_2:
                        all_team_players.extend(all_events[all_events['team'] == selected_team_2]['player'].dropna().unique())
                    all_team_players = list(set(all_team_players))  # Remove duplicates
                    
                    # Calculate stats for second player
                    player_stats_2 = calculate_player_radar_stats(
                        all_events, passes_euro, shots_euro, 
                        None, selected_team_2, selected_player_2
                    )
                else:
                    all_team_players = all_events[all_events['team'] == selected_team]['player'].dropna().unique()
                    player_stats_2 = None
                
                # Calculate stats for all players for proper normalization
                all_player_stats = {}
                for player in all_team_players:
                    # Determine which team this player belongs to
                    player_team = all_events[all_events['player'] == player]['team'].iloc[0] if not all_events[all_events['player'] == player].empty else None
                    if player_team:
                        stats = calculate_player_radar_stats(
                            all_events, passes_euro, shots_euro, 
                            None, player_team, player
                        )
                        if stats:
                            all_player_stats[player] = stats
                
                # Normalize stats
                normalized_stats = normalize_radar_stats(all_player_stats)
                selected_player_normalized = normalized_stats.get(selected_player, {})
                
                # Create radar chart (single or comparison)
                if compare_players and selected_player_2 and player_stats_2:
                    selected_player_2_normalized = normalized_stats.get(selected_player_2, {})
                    
                    # Use display names for chart
                    player_1_display = get_player_display_name(selected_player, lineups)
                    player_2_display = get_player_display_name(selected_player_2, lineups)
                    
                    fig = plot_player_radar_chart(
                        selected_player_normalized, player_1_display, theme,
                        player_2_stats=selected_player_2_normalized, 
                        player_2_name=player_2_display
                    )
                    chart_title = f"### {player_1_display} vs {player_2_display} - Tournament Performance"
                else:
                    player_1_display = get_player_display_name(selected_player, lineups)
                    fig = plot_player_radar_chart(selected_player_normalized, player_1_display, theme)
                    chart_title = f"### {player_1_display} - {selected_team} Tournament Performance"
                
                if fig:
                    st.markdown(chart_title)
                    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
                else:
                    st.error("Could not generate radar chart for this player.")
            else:
                st.error("No data available for this player in the tournament.")
        
        with col1:
            if player_stats:
                # Get display names for both players
                player_1_display = get_player_display_name(selected_player, lineups)
                
                if compare_players and selected_player_2 and player_stats_2:
                    player_2_display = get_player_display_name(selected_player_2, lineups)
                    st.markdown("### Tournament Statistics (Per 90)")
                    st.markdown(f"**{player_1_display}** ({selected_team}) **vs** **{player_2_display}** ({selected_team_2})")
                    st.metric("Total Minutes", f"{player_stats.get('minutes_played', 0):.0f} / {player_stats_2.get('minutes_played', 0):.0f}")
                    st.metric("Passes per 90", f"{player_stats.get('passes_per_90', 0):.1f} / {player_stats_2.get('passes_per_90', 0):.1f}")
                    st.metric("Pass Accuracy", f"{player_stats.get('pass_accuracy', 0):.1f}% / {player_stats_2.get('pass_accuracy', 0):.1f}%")
                    st.metric("Progressive Passes per 90", f"{player_stats.get('progressive_passes_per_90', 0):.1f} / {player_stats_2.get('progressive_passes_per_90', 0):.1f}")
                    st.metric("Key Passes per 90", f"{player_stats.get('key_passes_per_90', 0):.1f} / {player_stats_2.get('key_passes_per_90', 0):.1f}")
                    st.metric("Successful Dribbles per 90", f"{player_stats.get('dribbles_completed_per_90', 0):.1f} / {player_stats_2.get('dribbles_completed_per_90', 0):.1f}")
                    st.metric("Shots per 90", f"{player_stats.get('shots_per_90', 0):.1f} / {player_stats_2.get('shots_per_90', 0):.1f}")
                    st.metric("Goals per 90", f"{player_stats.get('goals_per_90', 0):.2f} / {player_stats_2.get('goals_per_90', 0):.2f}")
                    st.metric("Assists per 90", f"{player_stats.get('assists_per_90', 0):.2f} / {player_stats_2.get('assists_per_90', 0):.2f}")
                    st.metric("Non-Penalty G+A", f"{player_stats.get('non_penalty_g_a', 0)} ({player_stats.get('non_penalty_xg_xa', 0):.2f}) / {player_stats_2.get('non_penalty_g_a', 0)} ({player_stats_2.get('non_penalty_xg_xa', 0):.2f})")
                else:
                    st.markdown("### Tournament Statistics (Per 90)")
                    st.markdown(f"**{player_1_display}** ({selected_team})")
                    st.metric("Total Minutes", f"{player_stats.get('minutes_played', 0):.0f}")
                    st.metric("Passes per 90", f"{player_stats.get('passes_per_90', 0):.1f}")
                    st.metric("Pass Accuracy", f"{player_stats.get('pass_accuracy', 0):.1f}%")
                    st.metric("Progressive Passes per 90", f"{player_stats.get('progressive_passes_per_90', 0):.1f}")
                    st.metric("Key Passes per 90", f"{player_stats.get('key_passes_per_90', 0):.1f}")
                    st.metric("Successful Dribbles per 90", f"{player_stats.get('dribbles_completed_per_90', 0):.1f}")
                    st.metric("Shots per 90", f"{player_stats.get('shots_per_90', 0):.1f}")
                    st.metric("Goals per 90", f"{player_stats.get('goals_per_90', 0):.2f}")
                    st.metric("Assists per 90", f"{player_stats.get('assists_per_90', 0):.2f}")
                    st.metric("Non-Penalty G+A", f"{player_stats.get('non_penalty_g_a', 0)} ({player_stats.get('non_penalty_xg_xa', 0):.2f})")

        with col3:
            if player_stats:
                # Get display names for both players
                player_1_display = get_player_display_name(selected_player, lineups)
                
                if compare_players and selected_player_2 and player_stats_2:
                    player_2_display = get_player_display_name(selected_player_2, lineups)
                    st.markdown("### Defensive Actions (Per 90)")
                    st.markdown(f"**{player_1_display}** ({selected_team}) **vs** **{player_2_display}** ({selected_team_2})")
                    st.metric("Tackles per 90", f"{player_stats.get('tackles_per_90', 0):.1f} / {player_stats_2.get('tackles_per_90', 0):.1f}")
                    st.metric("Interceptions per 90", f"{player_stats.get('interceptions_per_90', 0):.1f} / {player_stats_2.get('interceptions_per_90', 0):.1f}")
                    st.metric("Recoveries per 90", f"{player_stats.get('recoveries_per_90', 0):.1f} / {player_stats_2.get('recoveries_per_90', 0):.1f}")
                    st.metric("Blocks per 90", f"{player_stats.get('blocks_per_90', 0):.1f} / {player_stats_2.get('blocks_per_90', 0):.1f}")
                    st.metric("Clearances per 90", f"{player_stats.get('clearances_per_90', 0):.1f} / {player_stats_2.get('clearances_per_90', 0):.1f}")
                    st.metric("50/50s Won per 90", f"{player_stats.get('50_50s_won_per_90', 0):.1f} / {player_stats_2.get('50_50s_won_per_90', 0):.1f}")
                else:
                    st.markdown("### Defensive Actions (Per 90)")
                    st.markdown(f"**{player_1_display}** ({selected_team})")
                    st.metric("Tackles per 90", f"{player_stats.get('tackles_per_90', 0):.1f}")
                    st.metric("Interceptions per 90", f"{player_stats.get('interceptions_per_90', 0):.1f}")
                    st.metric("Recoveries per 90", f"{player_stats.get('recoveries_per_90', 0):.1f}")
                    st.metric("Blocks per 90", f"{player_stats.get('blocks_per_90', 0):.1f}")
                    st.metric("Clearances per 90", f"{player_stats.get('clearances_per_90', 0):.1f}")
                    st.metric("50/50s Won per 90", f"{player_stats.get('50_50s_won_per_90', 0):.1f}")
    else:
        st.info("Please select a team and player from the sidebar to view the player analysis.")