import streamlit as st
import pandas as pd
from data_loader import (
    load_data, 
    calculate_xgchain, 
    prepare_shot_map_data, 
    prepare_pass_network_data,
    calculate_player_radar_stats,
    normalize_radar_stats,
    calculate_tournament_xgchain,
    prepare_tournament_pass_network_data
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
        
        # Get all teams
        all_teams = sorted(euro_groups['team'].dropna().unique())
        selected_team = st.selectbox("Select Team", all_teams)
        
        min_passes = st.slider("Minimum number of passes", 1, 50, 20)

    elif view_option == "Player Analysis":
        st.markdown("## Player Analysis Filters")
        
        # Get all teams
        all_teams = sorted(euro_groups['team'].dropna().unique())
        selected_team = st.selectbox("Select Team", all_teams)
        
        # Get all players for the selected team across all matches
        team_players = all_events[
            all_events['team'] == selected_team
        ]['player'].dropna().unique()
        
        if len(team_players) > 0:
            selected_player = st.selectbox("Select Player", sorted(team_players))
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
    player_counts = shots_euro['player'].value_counts().sort_values(ascending=False)
    player_list = ['All'] + player_counts.index.tolist()
    option = st.radio("Filter by:", ["Country", "Player"])
    if option == "Country":
        selected_country = st.selectbox("Select Country", country_list, index=0)
        selected_player = None
    else:
        selected_player = st.selectbox("Select Player", player_list, index=0)
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
    if selected_team:
        col1, col2, col3 = st.columns([1.5, 4, 0.5])

        with col2:
            # Calculate tournament-wide xGChain for the team
            tournament_xgchain_data = calculate_tournament_xgchain(all_events, selected_team)
            
            # Get total passes for the team across all matches
            total_passes = passes_euro[passes_euro['team'] == selected_team].shape[0]
            st.markdown(f"### {selected_team} — Tournament Performance — Total Passes: {total_passes}")
            
            fig, stats = plot_pass_network_plotly(
                passes_euro=passes_euro,
                lineups=lineups,
                euro_matches=euro_matches,
                match_id=None,  # None indicates tournament-wide
                team_name=selected_team,    
                min_passes=min_passes, 
                theme=theme,
                xgchain_data=tournament_xgchain_data,
                node_color=GOAL_COLOR,
                edge_color=MISS_COLOR
            )
            if fig:
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        
        with col1:
            st.markdown("### Key Player Stats")
            if stats:
                if 'max_centrality' in stats:
                    player, rank = stats['max_centrality']
                    st.metric(label="Highest Centrality", value=player, delta=f"Rank {rank}")
                if 'max_betweenness' in stats:
                    player, rank = stats['max_betweenness']
                    st.metric(label="Best Playmaker", value=player, delta=f"Rank {rank}")
                if 'most_progressive' in stats:
                    player, value = stats['most_progressive']
                    st.metric(label="Most Progressive Passes", value=player, delta=f"{int(value)} passes")
                if 'best_under_pressure' in stats:
                    player, rate, attempts = stats['best_under_pressure']
                    st.metric(label="Best Under Pressure (>5 attempts)", value=player, delta=f"{rate:.1f}% ({attempts} att)")
                if 'network_density' in stats:
                    density = stats['network_density']
                    st.metric(label="Network Density", value=f"{density:.1f}%", delta="Team Connectivity")
                if 'most_passes' in stats:
                    player, value = stats['most_passes']
                    st.metric(label="Most Passes Given", value=player, delta=f"{int(value)} passes")
                if 'most_received' in stats:
                    player, value = stats['most_received']
                    st.metric(label="Most Passes Received", value=player, delta=f"{int(value)} passes")
                if 'most_accurate' in stats:
                    player, value = stats['most_accurate']
                    st.metric(label="Highest Pass Accuracy (>50 attempts)", value=player, delta=f"{value:.1f}%")
            else:
                st.info("Not enough data to calculate player stats.")

        with col3:
            if tournament_xgchain_data:
                max_player = max(tournament_xgchain_data, key=tournament_xgchain_data.get)
                min_player = min(tournament_xgchain_data, key=tournament_xgchain_data.get)
                
                # Get player display names
                max_player_display = max_player.split()[-1] if max_player else max_player
                min_player_display = min_player.split()[-1] if min_player else min_player

                st.markdown(
                    f"<div style='display: flex; flex-direction: column; align-items: center; height: 100%; justify-content: space-between; margin-top: 60px;'>"
                    f"<div style='text-align: center; margin-bottom: 10px;'>"
                    f"<p style='margin: 0; font-weight: bold; font-size: 20px;'>Tournament xGChain</p>"
                    f"<p style='margin: 0; font-weight: bold;'>Max ({max(tournament_xgchain_data.values()):.2f})</p>"
                    f"<p style='margin: 0; font-size: 12px;'>{max_player_display}</p>"
                    f"</div>"
                    "<div style='width: 10px; height: 460px; background: linear-gradient(to bottom, rgba(253,231,37,1), rgba(53,183,121,1), rgba(49,104,142,1), rgba(68,1,84,1));'></div>"
                    f"<div style='text-align: center; margin-top: 10px;'>"
                    f"<p style='margin: 0; font-weight: bold;'>Min ({min(tournament_xgchain_data.values()):.2f})</p>"
                    f"<p style='margin: 0; font-size: 12px;'>{min_player_display}</p>"
                    f"</div>"
                    "</div>",
                    unsafe_allow_html=True
                )

    else:
        st.info("Please select a team from the sidebar to view the pass network.")

elif view_option == "Player Analysis":
    if selected_team and selected_player:
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            # Calculate tournament-wide player radar stats
            player_stats = calculate_player_radar_stats(
                all_events, passes_euro, shots_euro, 
                None, selected_team, selected_player  # None for tournament-wide
            )
            
            if player_stats:
                # Get all players from the team for normalization
                all_team_players = all_events[all_events['team'] == selected_team]['player'].dropna().unique()
                
                # Calculate stats for all players in the team for proper normalization
                all_player_stats = {}
                for player in all_team_players:
                    stats = calculate_player_radar_stats(
                        all_events, passes_euro, shots_euro, 
                        None, selected_team, player  # None for tournament-wide
                    )
                    if stats:
                        all_player_stats[player] = stats
                
                # Normalize stats
                normalized_stats = normalize_radar_stats(all_player_stats)
                selected_player_normalized = normalized_stats.get(selected_player, {})
                
                # Create radar chart
                fig = plot_player_radar_chart(selected_player_normalized, selected_player, theme)
                
                if fig:
                    st.markdown(f"### {selected_player} - {selected_team} Tournament Performance")
                    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
                else:
                    st.error("Could not generate radar chart for this player.")
            else:
                st.error("No data available for this player in the tournament.")
        
        with col1:
            if player_stats:
                st.markdown("### Tournament Statistics")
                st.metric("Total Minutes", f"{player_stats.get('minutes_played', 0):.0f}")
                st.metric("Total Passes", f"{player_stats.get('total_passes', 0)}")
                st.metric("Pass Accuracy", f"{player_stats.get('pass_accuracy', 0):.1f}%")
                st.metric("Progressive Passes", f"{player_stats.get('progressive_passes', 0)}")
                st.metric("Key Passes", f"{player_stats.get('key_passes', 0)}")
                st.metric("Total Shots", f"{player_stats.get('shots', 0)}")
                st.metric("Total Goals", f"{player_stats.get('goals', 0)}")
                st.metric("Total Assists", f"{player_stats.get('assists', 0)}")
        
        with col3:
            if player_stats:
                st.markdown("### Defensive Actions")
                st.metric("Tackles", f"{player_stats.get('tackles', 0)}")
                st.metric("Interceptions", f"{player_stats.get('interceptions', 0)}")
                st.metric("Recoveries", f"{player_stats.get('recoveries', 0)}")
                st.metric("Blocks", f"{player_stats.get('blocks', 0)}")
                st.metric("Clearances", f"{player_stats.get('clearances', 0)}")
                st.metric("50/50s Won", f"{player_stats.get('50_50s_won', 0)}")  # Replace "Duel success rate"
    else:
        st.info("Please select a team and player from the sidebar to view the player analysis.")