import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os


# --- Data Loading ---
euro_matches = pd.read_csv('data/euro2024_matches.csv')
euro_matches = euro_matches[['match_id', 'home_team', 'away_team']]

def get_opponent(row):
    if row['team'] == row['home_team']:
        return row['away_team']
    elif row['team'] == row['away_team']:
        return row['home_team']
    return None

events = pd.read_csv('data/euro2024_shots.csv')

shots_euro = events[events['type'] == 'Shot']
shots_euro = shots_euro.merge(euro_matches, on='match_id', how='left')
shots_euro['opponent_team'] = shots_euro.apply(get_opponent, axis=1)

penalties  = events[events['shot_type'] == 'Penalty']
penalties = penalties.merge(euro_matches, on='match_id', how='left')
penalties['opponent_team'] = penalties.apply(get_opponent, axis=1)

def plotly_penalty_map_center_only(
    shots_df: pd.DataFrame,
    *,
    outcome_col: str = 'shot_outcome',
    jitter: float = 0.03,
    palette: str = 'neutral',
    theme: str = 'green',
    title: str = '',
    shot_type_label: str = 'Shots',
):
    # Geometry and scaling
    depth_scale = 0.13
    width_scale = 2.4
    GOAL_Y_MIN, GOAL_Y_MAX = 36.0, 44.0
    GOAL_CENTER_Y, GOAL_Z_MAX = 40.0, 2.67
    GOAL_WIDTH = GOAL_Y_MAX - GOAL_Y_MIN
    SIX_LEFT, SIX_RIGHT = 33.0, 47.0
    PEN_LEFT, PEN_RIGHT = 18.0, 62.0
    penalty_spot_distance = 11.0
    six_yard_depth = 5.5
    penalty_area_depth = 16.5
    def z_m(d_m):
        return -d_m * depth_scale

    z_six = z_m(six_yard_depth)
    z_pen = z_m(penalty_area_depth)
    z_spot = z_m(penalty_spot_distance)
    # Horizontal view scaled to goal width
    span = GOAL_WIDTH * width_scale / 2
    x_left = GOAL_CENTER_Y - span
    x_right = GOAL_CENTER_Y + span
    # Vertical axis compressed
    y_bottom = z_pen - 0.5
    y_top = GOAL_Z_MAX + 1.2

    shots = shots_df.copy()
    if shots.empty:
        st.warning("No shots of this type were made for selected filter.")
        return None
    def extract_xyz(val):
        if isinstance(val, (list, tuple)) and len(val) == 3:
            return val
        if isinstance(val, str):
            try:
                arr = eval(val)
                if isinstance(arr, (list, tuple)) and len(arr) == 3:
                    return arr
            except Exception:
                pass
        return [np.nan, np.nan, np.nan]

    xyz = shots['shot_end_location'].apply(extract_xyz).tolist()
    shots[['end_x', 'end_y', 'end_z']] = pd.DataFrame(xyz, index=shots.index)
    shots['is_goal'] = shots[outcome_col].astype(str).eq('Goal')
    shots['plot_y'] = shots['end_y'] + np.random.uniform(-jitter, jitter, len(shots))
    shots['plot_z'] = shots['end_z'] + np.random.uniform(-jitter, jitter, len(shots))
    total = len(shots)
    goals = int(shots['is_goal'].sum())
    not_goals = total - goals
    goal_rate = 100 * goals / total if total else 0
    miss_types = shots.loc[~shots['is_goal'], outcome_col].value_counts().sort_values(ascending=False)
    body_part_counts = shots['shot_body_part'].value_counts()
    shootout_count = int((shots['period'] == 5).sum())
    regular_count = total - shootout_count
    # Plotly figure
    fig = go.Figure()
    # Theme colors
    t = theme.lower()
    if t == "white":
        PITCH_BG, LINE, NET, TEXT = "#f7f7f7", "#333333", "#aaaaaa", "#111111"
    elif t == "black":
        PITCH_BG, LINE, NET, TEXT = "#000000", "#e6e6e6", "#e6e6e6", "#e6e6e6"
    else:
        PITCH_BG, LINE, NET, TEXT = "#2d5e2e", "white", "white", "black"
    # Goal frame (left, right, top) with opacity 0.8
    fig.add_shape(type="line", x0=GOAL_Y_MIN, y0=0, x1=GOAL_Y_MIN, y1=GOAL_Z_MAX, line=dict(color=LINE, width=3), opacity=0.9)
    fig.add_shape(type="line", x0=GOAL_Y_MAX, y0=0, x1=GOAL_Y_MAX, y1=GOAL_Z_MAX, line=dict(color=LINE, width=3), opacity=0.9)
    fig.add_shape(type="line", x0=GOAL_Y_MIN, y0=GOAL_Z_MAX, x1=GOAL_Y_MAX, y1=GOAL_Z_MAX, line=dict(color=LINE, width=3), opacity=0.9)
    # Bottom line (goal line) with opacity 0.5
    fig.add_shape(type="line", x0=PEN_LEFT, y0=0, x1=PEN_RIGHT, y1=0, line=dict(color=LINE, width=3), opacity=0.7)
    # Net (vertical)
    net_spacing = 0.35
    for y in np.arange(GOAL_Y_MIN + net_spacing, GOAL_Y_MAX, net_spacing):
        fig.add_shape(type="line", x0=y, y0=0, x1=y, y1=GOAL_Z_MAX, line=dict(color=NET, width=1), opacity=0.3)
    # Net (horizontal)
    for z in np.arange(net_spacing, GOAL_Z_MAX, net_spacing):
        fig.add_shape(type="line", x0=GOAL_Y_MIN, y0=z, x1=GOAL_Y_MAX, y1=z, line=dict(color=NET, width=1), opacity=0.3)
    # 6-yard box
    fig.add_shape(type="line", x0=SIX_LEFT, y0=0, x1=SIX_LEFT, y1=z_six, line=dict(color=LINE, width=2), opacity=0.7)
    fig.add_shape(type="line", x0=SIX_RIGHT, y0=0, x1=SIX_RIGHT, y1=z_six, line=dict(color=LINE, width=2), opacity=0.7)
    fig.add_shape(type="line", x0=SIX_LEFT, y0=z_six, x1=SIX_RIGHT, y1=z_six, line=dict(color=LINE, width=2), opacity=0.7)
    # 18-yard box
    fig.add_shape(type="line", x0=PEN_LEFT, y0=0, x1=PEN_LEFT, y1=z_pen, line=dict(color=LINE, width=2), opacity=0.7)
    fig.add_shape(type="line", x0=PEN_RIGHT, y0=0, x1=PEN_RIGHT, y1=z_pen, line=dict(color=LINE, width=2), opacity=0.7)
    fig.add_shape(type="line", x0=PEN_LEFT, y0=z_pen, x1=PEN_RIGHT, y1=z_pen, line=dict(color=LINE, width=2), opacity=0.7)
    # Penalty spot
    fig.add_shape(type="line", x0=GOAL_CENTER_Y - 0.14, y0=z_spot, x1=GOAL_CENTER_Y + 0.14, y1=z_spot,
                  line=dict(color=LINE, width=3), opacity=0.4)
    # Scatter plot
    is_penalty = shots['shot_type'].eq('Penalty').all()
    if is_penalty:
        hover_text = shots.apply(lambda row: f"{row['player']}<br>{row['team']} vs {row['opponent_team']}<br>{'Shootout' if row['period']==5 else 'Regular Play'}<br>{row['shot_body_part']}<br>xG: {row.get('shot_statsbomb_xg', 0):.2f}", axis=1)
    else:
        def get_pattern_or_penalty(row):
            if row.get('shot_type', '') == 'Penalty':
                return 'Penalty'
            return row.get('play_pattern', '')
        hover_text = shots.apply(lambda row: f"{row['player']}<br>{row['team']} vs {row['opponent_team']}<br>{get_pattern_or_penalty(row)}<br>{row['shot_body_part']}<br>xG: {row.get('shot_statsbomb_xg', 0):.2f}", axis=1)
    if is_penalty:
        sizes = [12] * len(shots)
    else:
        min_size, max_size = 5, 13
        xg_values = shots['shot_statsbomb_xg'].fillna(0)
        xg_sqrt = np.sqrt(xg_values)
        sizes = min_size + (max_size - min_size) * (xg_sqrt - xg_sqrt.min()) / (xg_sqrt.max() - xg_sqrt.min() + 1e-6)
    fig.add_trace(go.Scatter(
        x=shots['plot_y'],
        y=shots['plot_z'],
        mode='markers',
        marker=dict(
            size=sizes,
            color=[GOAL_COLOR if g else MISS_COLOR for g in shots['is_goal']],
            line=dict(width=1, color=TEXT),  # Circle outline for each dot
            opacity=[0.95 if g else 0.25 for g in shots['is_goal']],
        ),
        text=hover_text,
        hoverinfo='text',
        name='Shots',
    ))

    # Add legend as annotations inside the graphic (top left corner)
    legend_x = x_left + 0.3
    legend_y1 = y_top - 0.3
    legend_y2 = legend_y1 - 0.45
    legend_y3 = legend_y2 - 0.45
    legend_font_color = "white" if t == "black" else "black"
    fig.add_annotation(
        x=legend_x, y=legend_y1,
        text=f"<span style='color:{GOAL_COLOR};font-weight:bold;font-size:22px'>●</span> <b>Goal</b>",
        showarrow=False,
        font=dict(family="DejaVu Sans Mono", size=16, color=legend_font_color),
        xanchor='left', yanchor='middle',
        align='left',
        bgcolor=None,
        borderpad=2,
        bordercolor=None,
        borderwidth=0,
        opacity=1,
    )
    fig.add_annotation(
        x=legend_x, y=legend_y2,
        text=f"<span style='color:{MISS_COLOR};font-weight:bold;font-size:22px'>●</span> <b>No Goal</b>",
        showarrow=False,
        font=dict(family="DejaVu Sans Mono", size=16, color=legend_font_color),
        xanchor='left', yanchor='middle',
        align='left',
        bgcolor=None,
        borderpad=2,
        bordercolor=None,
        borderwidth=0,
        opacity=1,
    )
    fig.add_annotation(
        x=legend_x,
        y=legend_y3,
        text="<span style='font-weight:bold'>Dot size = xG value</span>",
        showarrow=False,
        font=dict(family="DejaVu Sans Mono", size=15, color=legend_font_color),
        xanchor='left',
        yanchor='middle',
        align='left',
        bgcolor=None,
        borderpad=2,
        bordercolor=None,
        borderwidth=0,
        opacity=1,
    )

    # Calculate dynamic width and height based on geometry
    fig_width = int((x_right - x_left) * 300)
    fig_height = int((y_top - y_bottom) * 60)
    fig.update_layout(
        title=None,
        plot_bgcolor=PITCH_BG,
        paper_bgcolor=PITCH_BG,
        xaxis=dict(showgrid=False, zeroline=False, visible=False, range=[x_left, x_right]),
        yaxis=dict(showgrid=False, zeroline=False, visible=False, range=[y_bottom, y_top]),
        height=fig_height,
        width=fig_width,
        margin=dict(l=0, r=0, t=0, b=0),
    )
    # # Add arrow to highest xG miss if plotting all shots
    # if shot_type_label == 'Shots' and not penalties.empty:
    #     misses = penalties[~penalties['is_goal']].copy()
    #     if not misses.empty:
    #         idx_max_xg = misses['shot_statsbomb_xg'].idxmax()
    #         miss_row = misses.loc[idx_max_xg]
    #         arrow_x = miss_row['plot_y']
    #         arrow_y = miss_row['plot_z']
    #         text_y = y_bottom + 2.2
    #         fig.add_annotation(
    #             x=arrow_x, y=text_y,
    #             ax=arrow_x, ay=arrow_y,
    #             text="Highest xG Miss",
    #             showarrow=True,
    #             arrowhead=2,
    #             arrowsize=2,
    #             arrowwidth=2.5,
    #             arrowcolor="#d62728",
    #             font=dict(size=15, color="#d62728"),
    #             borderpad=2,
    #             opacity=1,
    #         )

    return fig, total, goals, not_goals, goal_rate, miss_types, body_part_counts, is_penalty, shootout_count, regular_count

# --- Streamlit UI ---
# st.title("Euro 2024 Shot Map")

with st.sidebar:
    st.markdown("# Euro 2024 Shot Map")
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
    selected_miss_type = st.selectbox("Miss Type", miss_type_options, index=0)

    theme = st.selectbox("Theme", ["green", "white", "black"])

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

# Assign palette and colors based on theme
if theme == "green":
    palette = "neutral"
    GOAL_COLOR = "#222222"
    MISS_COLOR = "#bbbbbb"
elif theme == "black":
    palette = "classic"
    GOAL_COLOR = "#2ca02c"
    MISS_COLOR = "#d62728"
else:
    palette = "vibrant"
    GOAL_COLOR = "#1f77b4"
    MISS_COLOR = "#ff7f0e"

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
fig_data = plotly_penalty_map_center_only(filtered_shots, theme=theme, palette=palette, title=title, shot_type_label=shot_type_label)
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
                penalty_count = shots_euro[shots_euro['shot_type'] == 'Penalty'].shape[0]
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