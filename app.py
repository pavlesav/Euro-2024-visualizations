import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os

# --- Data Loading ---
events = pd.read_csv('data/euro2024_shots.csv')
shots_euro = events[events['type'] == 'Shot']
penalties  = events[events['shot_type'] == 'Penalty']
euro_matches = pd.read_csv('data/euro2024_matches.csv')

matches = euro_matches[['match_id', 'home_team', 'away_team']]
def get_opponent(row):
    if row['team'] == row['home_team']:
        return row['away_team']
    elif row['team'] == row['away_team']:
        return row['home_team']
    return None
penalties = penalties.merge(matches, on='match_id', how='left')
penalties['opponent_team'] = penalties.apply(get_opponent, axis=1)

# --- Plotly Penalty Map ---
def plotly_penalty_map(
    shots_df: pd.DataFrame,
    *,
    outcome_col: str = 'shot_outcome',
    jitter: float = 0.04,
    palette: str = 'neutral',
    theme: str = 'green',
    title: str = '',
):
    # Geometry and scaling 
    depth_scale = 0.2
    width_scale = 3
    GOAL_Y_MIN, GOAL_Y_MAX = 36.0, 44.0
    GOAL_CENTER_Y, GOAL_Z_MAX = 40.0, 2.67
    GOAL_WIDTH = GOAL_Y_MAX - GOAL_Y_MIN
    SIX_LEFT, SIX_RIGHT = 30.0, 50.0
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
    y_top = GOAL_Z_MAX + 0.6
    # Theme colors
    t = theme.lower()
    if t == "white":
        PITCH_BG, LINE, NET, TEXT = "#f7f7f7", "#333333", "#aaaaaa", "#111111"
    elif t == "black":
        PITCH_BG, LINE, NET, TEXT = "#000000", "#e6e6e6", "#e6e6e6", "#e6e6e6"
    else:
        PITCH_BG, LINE, NET, TEXT = "#2d5e2e", "white", "white", "black"
    if palette == 'vibrant':
        GOAL_COLOR = "#1f77b4"
        MISS_COLOR = "#ff7f0e"
    elif palette == 'classic':
        GOAL_COLOR = "#2ca02c"
        MISS_COLOR = "#d62728"
    else:
        GOAL_COLOR = "#222222"
        MISS_COLOR = "#bbbbbb"
    # Prepare data
    penalties = shots_df.copy()
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
    xyz = penalties['shot_end_location'].apply(extract_xyz).tolist()
    penalties[['end_x', 'end_y', 'end_z']] = pd.DataFrame(xyz, index=penalties.index)
    penalties['is_goal'] = penalties[outcome_col].astype(str).eq('Goal')
    # Jitter
    penalties['plot_y'] = penalties['end_y'] + np.random.uniform(-jitter, jitter, len(penalties))
    penalties['plot_z'] = penalties['end_z'] + np.random.uniform(-jitter, jitter, len(penalties))
    # KPIs and legend
    total = len(penalties)
    goals = int(penalties['is_goal'].sum())
    not_goals = total - goals
    goal_rate = 100 * goals / total if total else 0
    miss_types = penalties.loc[~penalties['is_goal'], outcome_col].value_counts().sort_index()
    body_part_counts = penalties['shot_body_part'].value_counts()
    right_foot = int(body_part_counts.get('Right Foot', 0))
    left_foot  = int(body_part_counts.get('Left Foot', 0))
    shootout_count = int((penalties['period'] == 5).sum())
    regular_count = total - shootout_count
    # Plotly figure
    fig = go.Figure()
    # Goal frame
    fig.add_shape(type="rect", x0=GOAL_Y_MIN, y0=0, x1=GOAL_Y_MAX, y1=GOAL_Z_MAX,
                  line=dict(color=LINE, width=3), fillcolor=None, layer="above")
    # Goal line
    fig.add_shape(type="line", x0=PEN_LEFT, y0=0, x1=PEN_RIGHT, y1=0, line=dict(color=LINE, width=2))
    # Net (vertical)
    net_spacing = 0.35
    net_alpha = 0.18
    for y in np.arange(GOAL_Y_MIN + net_spacing, GOAL_Y_MAX, net_spacing):
        fig.add_shape(type="line", x0=y, y0=0, x1=y, y1=GOAL_Z_MAX, line=dict(color=NET, width=1), opacity=net_alpha)
    # Net (horizontal)
    for z in np.arange(net_spacing, GOAL_Z_MAX, net_spacing):
        fig.add_shape(type="line", x0=GOAL_Y_MIN, y0=z, x1=GOAL_Y_MAX, y1=z, line=dict(color=NET, width=1), opacity=net_alpha)
    # 6-yard box
    fig.add_shape(type="line", x0=SIX_LEFT, y0=0, x1=SIX_LEFT, y1=z_six, line=dict(color=LINE, width=2))
    fig.add_shape(type="line", x0=SIX_RIGHT, y0=0, x1=SIX_RIGHT, y1=z_six, line=dict(color=LINE, width=2))
    fig.add_shape(type="line", x0=SIX_LEFT, y0=z_six, x1=SIX_RIGHT, y1=z_six, line=dict(color=LINE, width=2))
    # 18-yard box
    fig.add_shape(type="line", x0=PEN_LEFT, y0=0, x1=PEN_LEFT, y1=z_pen, line=dict(color=LINE, width=2))
    fig.add_shape(type="line", x0=PEN_RIGHT, y0=0, x1=PEN_RIGHT, y1=z_pen, line=dict(color=LINE, width=2))
    fig.add_shape(type="line", x0=PEN_LEFT, y0=z_pen, x1=PEN_RIGHT, y1=z_pen, line=dict(color=LINE, width=2))
    # Penalty spot
    fig.add_shape(type="line", x0=GOAL_CENTER_Y - 0.14, y0=z_spot, x1=GOAL_CENTER_Y + 0.14, y1=z_spot,
                  line=dict(color=LINE, width=3))
    # Penalty points
    hover_text = penalties.apply(lambda row: f"{row['player']}<br>{row['team']} vs {row['opponent_team']}<br>{'Shootout' if row['period']==5 else 'Regular Play'}", axis=1)
    fig.add_trace(go.Scatter(
        x=penalties['plot_y'],
        y=penalties['plot_z'],
        mode='markers',
        marker=dict(
            size=14,
            color=[GOAL_COLOR if g else MISS_COLOR for g in penalties['is_goal']],
            line=dict(width=1.5, color=TEXT),
            opacity=0.9,
        ),
        text=hover_text,
        hoverinfo='text',
        name='Penalties',
    ))
    # Legend box coordinates (left of goal, shifted left)
    legend_box_width = 3
    legend_box_height = 1
    legend_x0 = x_left + 0.5
    legend_x1 = legend_x0 + legend_box_width
    legend_y0 = GOAL_Z_MAX - 1
    legend_y1 = legend_y0 + legend_box_height
    # --- KPI and legend annotations ---
    kx = 44.6  # right of goal
    ky = GOAL_Z_MAX + 0.9
    font = dict(family="DejaVu Sans Mono", size=16, color=TEXT)
    bold_font = dict(family="DejaVu Sans Mono", size=20, color=TEXT)
    annotations = [
        dict(x=kx, y=ky, text=f"<b>Total Penalties: {total}</b>", showarrow=False, font=bold_font, xanchor='left', yanchor='top'),
        dict(x=kx, y=ky-0.35, text=f"<b>Penalties Scored: {goals}</b>", showarrow=False, font=bold_font, xanchor='left', yanchor='top'),
        dict(x=kx, y=ky-0.7, text=f"<b>{total} Total Penalties</b>", showarrow=False, font=bold_font, xanchor='left', yanchor='top'),
        dict(x=kx, y=ky-1.2, text=f"<span style='color:{GOAL_COLOR}'><b>{goals} Scored</b> ({goal_rate:.1f}%)</span>", showarrow=False, font=bold_font, xanchor='left', yanchor='top'),
        dict(x=kx, y=ky-1.7, text=f"<span style='color:{MISS_COLOR}'><b>{not_goals} Missed/Saved</b></span>", showarrow=False, font=bold_font, xanchor='left', yanchor='top')
    ]
    miss_y = ky-2.15
    if not miss_types.empty:
        for miss_type, count in miss_types.items():
            annotations.append(dict(x=kx+0.5, y=miss_y, text=f"<span style='color:{MISS_COLOR}'>• {count} {miss_type}</span>", showarrow=False, font=font, xanchor='left', yanchor='top'))
            miss_y -= 0.4
    # Legend box (darker)
    fig.add_shape(type="rect", x0=legend_x0, y0=legend_y0, x1=legend_x1, y1=legend_y1,
                  line=dict(color="#333", width=2), fillcolor="#7a8b6a", layer="above")
    # Legend annotations inside box (left aligned)
    legend_text_y1 = legend_y1 - 0.3
    legend_text_y2 = legend_text_y1 - 0.45
    fig.add_annotation(x=legend_x0 + 0.10, y=legend_text_y1,
        text=f"<span style='color:{GOAL_COLOR};font-weight:bold'>● Goal</span>", showarrow=False,
        font=dict(family="DejaVu Sans Mono", size=16, color=TEXT), xanchor='left', yanchor='middle')
    fig.add_annotation(x=legend_x0 + 0.10, y=legend_text_y2,
        text=f"<span style='color:{MISS_COLOR};font-weight:bold'>● No Goal</span>", showarrow=False,
        font=dict(family="DejaVu Sans Mono", size=16, color=TEXT), xanchor='left', yanchor='middle')
    # Move details info below legend, left aligned, no 'Details' label
    details_y = legend_y0 - 0.3
    line1 = f"Regular Play: {regular_count} | Shootout: {shootout_count}"
    line2 = f"Right Foot: {right_foot} | Left Foot: {left_foot}"
    annotations.append(dict(x=legend_x0, y=details_y, text=line1, showarrow=False, font=font, xanchor='left', yanchor='top'))
    details_y -= 0.4
    annotations.append(dict(x=legend_x0, y=details_y, text=line2, showarrow=False, font=font, xanchor='left', yanchor='top'))
    fig.update_layout(annotations=annotations)
    # Layout: match notebook proportions
    # Title color based on theme
    title_font_color = TEXT
    fig.update_layout(
        title=title,
        title_font=dict(family="DejaVu Sans Mono", size=20, color=title_font_color),
        plot_bgcolor=PITCH_BG,
        paper_bgcolor=PITCH_BG,
        xaxis=dict(showgrid=False, zeroline=False, visible=False, range=[x_left, x_right]),
        yaxis=dict(showgrid=False, zeroline=False, visible=False, range=[y_bottom, y_top]),
        height=360,
        width=1100,
        margin=dict(l=20, r=20, t=60, b=20),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

# --- Streamlit UI ---
st.title("Euro 2024 Penalty Shot Map")
st.markdown("Select a country or player to view their penalty shot map.")
country_list = ['All'] + sorted(penalties['team'].dropna().unique())
player_list = ['All'] + sorted(penalties['player'].dropna().unique())

theme = st.sidebar.selectbox("Theme", ["green", "white", "black"])
if theme == "green":
    palette = "neutral"
elif theme == "black":
    palette = "classic"
else:
    palette = "vibrant"
option = st.radio("Filter by:", ["Country", "Player"])
if option == "Country":
    country = st.selectbox("Select Country", country_list, index=0)
    if country == 'All':
        plotly_penalty_map(penalties, theme=theme, palette=palette, title="All Euro 2024 Penalties")
    else:
        plotly_penalty_map(penalties[penalties['team'] == country], theme=theme, palette=palette, title=f"Penalty Analysis for {country}")
elif option == "Player":
    player = st.selectbox("Select Player", player_list, index=0)
    if player == 'All':
        plotly_penalty_map(penalties, theme=theme, palette=palette, title="All Euro 2024 Penalties")
    else:
        plotly_penalty_map(penalties[penalties['player'] == player], theme=theme, palette=palette, title=f"Penalty Analysis for {player}")