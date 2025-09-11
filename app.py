import streamlit as st
import pandas as pd
from data_loader import load_data, calculate_xgchain
from visualizations import plotly_penalty_map_center_only, plot_pass_network_plotly

# --- Data Loading ---
st.set_page_config(layout="wide")

shots_euro, euro_matches, passes_euro, lineups, euro_groups, all_events = load_data()

# --- Streamlit UI ---
with st.sidebar:
    st.markdown("# Euro 2024 Analysis")
    
    view_option = st.radio("Select View", ["Shot Map", "Pass Network"])
    
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
        
        # Hierarchical selection for matches
        stage_order = ['Final', 'Semi-finals', 'Quarter-finals', 'Round of 16', 'Group Stage']
        stages = euro_matches['competition_stage'].unique()
        sorted_stages = [s for s in stage_order if s in stages]
        
        selected_stage = st.selectbox("Select Stage", sorted_stages)

        matches_in_stage = euro_matches[euro_matches['competition_stage'] == selected_stage]

        if selected_stage == 'Group Stage':
            groups = sorted(euro_groups['group'].dropna().unique())
            selected_group = st.selectbox("Select Group", groups)
            
            # Get teams in the selected group
            teams_in_group = euro_groups[euro_groups['group'] == selected_group]['team'].tolist()
            
            # Filter matches where both teams are in the selected group
            matches_to_display = matches_in_stage[
                matches_in_stage['home_team'].isin(teams_in_group) & 
                matches_in_stage['away_team'].isin(teams_in_group)
            ]
        else:
            matches_to_display = matches_in_stage

        match_display_names = matches_to_display.apply(lambda row: f"{row['home_team']} vs {row['away_team']}", axis=1)
        match_map = {name: mid for name, mid in zip(match_display_names, matches_to_display['match_id'])}

        selected_match_name = st.selectbox("Select Match", match_display_names)
        
        if selected_match_name:
            match_id = match_map[selected_match_name]
            
            selected_match_info = euro_matches[euro_matches['match_id'] == match_id].iloc[0]
            team_options = [selected_match_info['home_team'], selected_match_info['away_team']]
            selected_team = st.radio("Select Team", team_options)

            min_passes = st.slider("Minimum number of passes", 1, 30, 10)
        else:
            match_id, selected_team, min_passes = None, None, None


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
    if match_id and selected_team:
        col1, col2, col3 = st.columns([1.5, 4, 0.5])

        with col2:
            xgchain_data = calculate_xgchain(all_events, match_id, selected_team)
            opponent_team = None
            match_row = euro_matches[euro_matches['match_id'] == match_id]
            if not match_row.empty:
                row = match_row.iloc[0]
                opponent_team = row['away_team'] if row['home_team'] == selected_team else row['home_team']
            total_passes = passes_euro[(passes_euro['match_id'] == match_id) & (passes_euro['team'] == selected_team)].shape[0]
            st.markdown(f"### {selected_team} vs {opponent_team} — Total Passes: {total_passes}")
            
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
            if xgchain_data:
                max_player = max(xgchain_data, key=xgchain_data.get)
                min_player = min(xgchain_data, key=xgchain_data.get)
                
                # Safely get nicknames
                max_player_nick_series = lineups.loc[(lineups['player_name'] == max_player) & (lineups['match_id'] == match_id), 'player_nickname']
                max_player_display = max_player_nick_series.iloc[0] if not max_player_nick_series.isnull().all() else max_player
                
                min_player_nick_series = lineups.loc[(lineups['player_name'] == min_player) & (lineups['match_id'] == match_id), 'player_nickname']
                min_player_display = min_player_nick_series.iloc[0] if not min_player_nick_series.isnull().all() else min_player

                st.markdown(
                    f"<div style='display: flex; flex-direction: column; align-items: center; height: 100%; justify-content: space-between; margin-top: 60px;'>"
                    f"<div style='text-align: center; margin-bottom: 10px;'>"
                    f"<p style='margin: 0; font-weight: bold; font-size: 20px;'>xGChain</p>"
                    f"<p style='margin: 0; font-weight: bold;'>Max ({max(xgchain_data.values()):.2f})</p>"
                    f"<p style='margin: 0; font-size: 12px;'>{max_player_display}</p>"
                    f"</div>"
                    "<div style='width: 10px; height: 460px; background: linear-gradient(to bottom, rgba(253,231,37,1), rgba(53,183,121,1), rgba(49,104,142,1), rgba(68,1,84,1));'></div>"
                    f"<div style='text-align: center; margin-top: 10px;'>"
                    f"<p style='margin: 0; font-weight: bold;'>Min ({min(xgchain_data.values()):.2f})</p>"
                    f"<p style='margin: 0; font-size: 12px;'>{min_player_display}</p>"
                    f"</div>"
                    "</div>",
                    unsafe_allow_html=True
                )

    else:
        st.info("Please select a match from the sidebar to view the pass network.")