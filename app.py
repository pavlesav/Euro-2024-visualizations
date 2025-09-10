import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
import networkx as nx
import glob

# --- Data Loading ---
st.set_page_config(layout="wide")

@st.cache_data
def load_data():
    shots = pd.read_csv('data/euro2024_shots.csv')
    matches = pd.read_csv('data/euro2024_matches.csv')
    passes = pd.read_csv('data/euro2024_passes.csv')
    lineups = pd.read_csv('data/lineups.csv')
    groups = pd.read_csv('data/euro2024_groups.csv')

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
    return shots_euro, matches, passes, lineups, groups

shots_euro, euro_matches, passes_euro, lineups, euro_groups = load_data()

def load_all_events(data_folder='data/'):
    # Load and concatenate all event CSV files in the main data folder (not subfolder)
    all_files = glob.glob(os.path.join(data_folder, '*.csv'))
    df_list = [pd.read_csv(f) for f in all_files if 'events' not in f and 'matches' not in f and 'groups' not in f and 'lineups' not in f and 'shots' not in f and 'passes' not in f]
    events = pd.concat(df_list, ignore_index=True)
    return events

# Load all events for xGChain calculation
all_events = load_all_events()

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
            xg = shot_rows['shot_statsbomb_xg'].max() if 'shot_statsbomb_xg' in shot_rows else 0
            players_in_chain = chain['player'].dropna().unique()
            for player in players_in_chain:
                xgchain[player] = xgchain.get(player, 0) + xg
    return xgchain

# --- Visualization Functions ---

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

def draw_full_pitch_plotly(
    *,
    theme: str = "green",
    title: str | None = None,
    title_font_size: int = 24,
    fig_width: int = 800,
    fig_height: int = 600,
):
    """
    Draws a full soccer pitch in Plotly based on StatsBomb coordinates (120x80).
    """
    t = theme.lower()
    if t == "white":
        PITCH_BG, LINE, TEXT = "#f7f7f7", "#333333", "#111111"
    elif t == "black":
        PITCH_BG, LINE, TEXT = "#000000", "#e6e6e6", "#e6e6e6"
    else:
        PITCH_BG, LINE, TEXT = "#2d5e2e", "white", "black"

    fig = go.Figure()
    lw = 1.5

    # Pitch outline
    fig.add_shape(type="rect", x0=0, y0=0, x1=120, y1=80, line=dict(color=LINE, width=lw), fillcolor=PITCH_BG)
    # Center line
    fig.add_shape(type="line", x0=60, y0=0, x1=60, y1=80, line=dict(color=LINE, width=lw))
    # Center circle
    fig.add_shape(type="circle", x0=50, y0=30, x1=70, y1=50, line=dict(color=LINE, width=lw))
    fig.add_shape(type="circle", x0=59.5, y0=39.5, x1=60.5, y1=40.5, line=dict(color=LINE, width=lw), fillcolor=LINE)
    # Penalty areas
    fig.add_shape(type="rect", x0=0, y0=18, x1=18, y1=62, line=dict(color=LINE, width=lw))
    fig.add_shape(type="rect", x0=102, y0=18, x1=120, y1=62, line=dict(color=LINE, width=lw))
    # 6-yard boxes
    fig.add_shape(type="rect", x0=0, y0=30, x1=6, y1=50, line=dict(color=LINE, width=lw))
    fig.add_shape(type="rect", x0=114, y0=30, x1=120, y1=50, line=dict(color=LINE, width=lw))
    # Penalty spots
    fig.add_shape(type="circle", x0=11.5, y0=39.5, x1=12.5, y1=40.5, line=dict(color=LINE, width=lw), fillcolor=LINE)
    fig.add_shape(type="circle", x0=107.5, y0=39.5, x1=108.5, y1=40.5, line=dict(color=LINE, width=lw), fillcolor=LINE)
    # Arcs
    fig.add_shape(type="path", path="M 18,31.1 C 23.3,35.5 23.3,44.5 18,48.9", line_color=LINE, line_width=lw)
    fig.add_shape(type="path", path="M 102,31.1 C 96.7,35.5 96.7,44.5 102,48.9", line_color=LINE, line_width=lw)
    # Goals
    fig.add_shape(type="rect", x0=-1.5, y0=36, x1=0, y1=44, line=dict(color=LINE, width=lw+1), fillcolor=PITCH_BG)
    fig.add_shape(type="rect", x0=120, y0=36, x1=121.5, y1=44, line=dict(color=LINE, width=lw+1), fillcolor=PITCH_BG)

    fig.update_layout(
        xaxis=dict(range=[-2, 122], showgrid=False, zeroline=False, visible=False),
        yaxis=dict(range=[-2, 82], showgrid=False, zeroline=False, visible=False),
        plot_bgcolor=PITCH_BG,
        paper_bgcolor=PITCH_BG,
        height=fig_height,
        width=fig_width,
        margin=dict(l=0, r=0, t=0, b=0),
    )
    # Remove title from graph
    return fig

def plot_pass_network_plotly(match_id, team_name, min_passes, theme, node_color, edge_color):
    """
    Generates and plots an interactive pass network using Plotly.
    """
    # Set hover background color and font color for theme
    if theme == "green":
        hover_bg = "black"
        hover_font = "white"
    elif theme == "black":
        hover_bg = "#222222"
        hover_font = "white"
    else:
        hover_bg = "white"
        hover_font = "black"

    def offset_edge_coords(x1, y1, x2, y2, offset_perp=0, offset_along=2.0):
        vec = np.array([x2 - x1, y2 - y1])
        norm_vec = vec / np.linalg.norm(vec) if np.linalg.norm(vec) > 0 else vec
        perp_vec = np.array([-norm_vec[1], norm_vec[0]]) * offset_perp
        # Move start/end a bit along the direction to avoid overlapping nodes
        start_offset = np.array([x1, y1]) + norm_vec * offset_along + perp_vec
        end_offset = np.array([x2, y2]) - norm_vec * offset_along + perp_vec
        return start_offset[0], start_offset[1], end_offset[0], end_offset[1]

    team_passes = passes_euro[(passes_euro['match_id'] == match_id) & (passes_euro['team'] == team_name)]
    if team_passes.empty:
        st.warning(f"No pass data found for {team_name} in this match.")
        return None

    starting_11 = team_passes.sort_values('minute').drop_duplicates('player').head(11)['player'].tolist()

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

    pass_counts = {}
    for _, row in team_passes.iterrows():
        passer = row['player']
        recipient = row['pass_recipient']
        if pd.notnull(passer) and pd.notnull(recipient) and passer in starting_11 and recipient in starting_11:
            key = (passer, recipient)
            pass_counts[key] = pass_counts.get(key, 0) + 1

    if lineups.empty:
        st.error("Lineup data is not available. Cannot generate pass network.")
        return None
        
    team_lineup = lineups[(lineups['match_id'] == match_id) & (lineups['team_name'] == team_name)]
    if team_lineup.empty:
        st.warning(f"No lineup data found for {team_name} in this match.")
        team_numbers = {p: p for p in starting_11}
        team_nicknames = {p: p.split()[-1] if p else "" for p in starting_11}
    else:
        team_numbers = {p['player_name']: str(p['jersey_number']) for _, p in team_lineup.iterrows()}
        team_nicknames = {
            p['player_name']: p['player_nickname'] if pd.notnull(p['player_nickname']) else p['player_name']
            for _, p in team_lineup.iterrows()
        }

    player_positions_num = {team_numbers.get(p, p): pos for p, pos in player_positions.items()}
    pass_counts_num = {(team_numbers.get(p1, p1), team_numbers.get(p2, p2)): count for (p1, p2), count in pass_counts.items()}
    
    jersey_to_display_name = {str(j): n for n, j in zip(team_nicknames.values(), team_numbers.values())}
    
    pass_counts_num_filtered = {k: v for k, v in pass_counts_num.items() if v >= min_passes}

    G = nx.DiGraph()
    for (u, v), weight in pass_counts_num_filtered.items():
        G.add_edge(u, v, weight=weight)
    
    degrees = dict(G.degree())

    opponent_team = None
    match_row = euro_matches[euro_matches['match_id'] == match_id]
    if not match_row.empty:
        row = match_row.iloc[0]
        opponent_team = row['away_team'] if row['home_team'] == team_name else row['home_team']
    total_passes = passes_euro[(passes_euro['match_id'] == match_id) & (passes_euro['team'] == team_name)].shape[0]
    title = f"{team_name} vs {opponent_team} — Total Passes: {total_passes}"
    fig = draw_full_pitch_plotly(theme=theme, title=title, title_font_size=28, fig_width=800, fig_height=600)
    
    nodes_data = list(player_positions_num.items())
    node_jersey_numbers = [item[0] for item in nodes_data]
    node_positions = [item[1] for item in nodes_data]
    node_x = [p[0] for p in node_positions]
    node_y = [p[1] for p in node_positions]

    node_sizes = [5 + degrees.get(node, 0) * 3 for node in node_jersey_numbers]
    
    node_edge_color = "black" if theme != "black" else "white"
    text_color = "white" if theme == "green" else "black"

    # Only one edge-drawing loop
    def offset_edge_coords(x1, y1, x2, y2, offset_perp=0, offset_along=2.0):
        vec = np.array([x2 - x1, y2 - y1])
        norm_vec = vec / np.linalg.norm(vec) if np.linalg.norm(vec) > 0 else vec
        perp_vec = np.array([-norm_vec[1], norm_vec[0]]) * offset_perp
        # Move start/end a bit along the direction to avoid overlapping nodes
        start_offset = np.array([x1, y1]) + norm_vec * offset_along + perp_vec
        end_offset = np.array([x2, y2]) - norm_vec * offset_along + perp_vec
        return start_offset[0], start_offset[1], end_offset[0], end_offset[1]

    bidirectional_drawn = set()
    edge_hover_traces = []
    for (u, v), weight in pass_counts_num_filtered.items():
        if (u, v) in bidirectional_drawn:
            continue
        x1, y1 = player_positions_num.get(str(u), (0, 0))
        x2, y2 = player_positions_num.get(str(v), (0, 0))
        is_bidirectional = (v, u) in pass_counts_num_filtered
        label_uv = f"{jersey_to_display_name.get(str(u), str(u))} → {jersey_to_display_name.get(str(v), str(v))}: {weight} passes"
        if is_bidirectional:
            x1_uv, y1_uv, x2_uv, y2_uv = offset_edge_coords(x1, y1, x2, y2, offset_perp=0.8, offset_along=3.0)
            fig.add_annotation(
                x=x2_uv, y=y2_uv, ax=x1_uv, ay=y1_uv,
                xref='x', yref='y', axref='x', ayref='y',
                showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=weight / 4,
                arrowcolor=edge_color, opacity=0.7,
                text=None
            )
            edge_hover_traces.append(go.Scatter(
                x=[x2_uv], y=[y2_uv],
                mode='markers',
                marker=dict(size=18, color='rgba(0,0,0,0)'),
                hoverinfo='text',
                hovertext=label_uv,
                showlegend=False,
                hoverlabel=dict(bgcolor=hover_bg, font=dict(color=hover_font, size=14))
            ))
            weight_vu = pass_counts_num_filtered.get((v, u), 0)
            label_vu = f"{jersey_to_display_name.get(str(v), str(v))} → {jersey_to_display_name.get(str(u), str(u))}: {weight_vu} passes"
            x1_vu, y1_vu, x2_vu, y2_vu = offset_edge_coords(x2, y2, x1, y1, offset_perp=0.8, offset_along=3.0)
            fig.add_annotation(
                x=x2_vu, y=y2_vu, ax=x1_vu, ay=y1_vu,
                xref='x', yref='y', axref='x', ayref='y',
                showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=weight_vu / 4,
                arrowcolor=edge_color, opacity=0.7,
                text=None
            )
            edge_hover_traces.append(go.Scatter(
                x=[x2_vu], y=[y2_vu],
                mode='markers',
                marker=dict(size=18, color='rgba(0,0,0,0)'),
                hoverinfo='text',
                hovertext=label_vu,
                showlegend=False,
                hoverlabel=dict(bgcolor=hover_bg, font=dict(color=hover_font, size=14))
            ))
            bidirectional_drawn.add((u, v)); bidirectional_drawn.add((v, u))
        else:
            x1_offset, y1_offset, x2_offset, y2_offset = offset_edge_coords(x1, y1, x2, y2, offset_perp=0, offset_along=3.0)
            fig.add_annotation(
                x=x2_offset, y=y2_offset, ax=x1_offset, ay=y1_offset,
                xref='x', yref='y', axref='x', ayref='y',
                showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=weight / 4,
                arrowcolor=edge_color, opacity=0.7,
                text=None
            )
            edge_hover_traces.append(go.Scatter(
                x=[x2_offset], y=[y2_offset],
                mode='markers',
                marker=dict(size=18, color='rgba(0,0,0,0)'),
                hoverinfo='text',
                hovertext=label_uv,
                showlegend=False,
                hoverlabel=dict(bgcolor=hover_bg, font=dict(color=hover_font, size=14))
            ))
            bidirectional_drawn.add((u, v))
    # Add invisible edge hover traces
    for trace in edge_hover_traces:
        fig.add_trace(trace)

    fig.update_layout(
        shapes=[dict(layer='below') for shape in fig.layout.shapes]
    )

    # Map player name to jersey number
    name_to_jersey = {name: jersey for name, jersey in team_numbers.items()}
    # Count passes made by each jersey number
    passes_by_jersey = {}
    for _, row in team_passes.iterrows():
        jersey = name_to_jersey.get(row['player'])
        if jersey:
            passes_by_jersey[jersey] = passes_by_jersey.get(jersey, 0) + 1

    # Calculate raw total passes made by each player in the match
    player_pass_counts = team_passes['player'].value_counts().to_dict()

    # Node hover traces
    hover_text = [f"#{j} {jersey_to_display_name.get(j, 'Unknown')}<br>Total Passes: {passes_by_jersey.get(j, 0)}" for j in node_jersey_numbers]
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=node_jersey_numbers,
        hovertext=hover_text,
        hoverinfo='text',
        marker=dict(
            color=node_color,
            size=[size + 15 for size in node_sizes],
            line=dict(width=2, color='black'),
            opacity=1.0
        ),
        textfont=dict(
            color='black',
            size=14,
            family="Arial, sans-serif"
        ),
        name='Players',
        hoverlabel=dict(bgcolor=hover_bg, font=dict(color=hover_font, size=14))
    ))

    fig.update_layout(showlegend=False)
    return fig


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

            min_passes = st.slider("Minimum number of passes", 1, 30, 5)
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
        opponent_team = None
        match_row = euro_matches[euro_matches['match_id'] == match_id]
        if not match_row.empty:
            row = match_row.iloc[0]
            opponent_team = row['away_team'] if row['home_team'] == selected_team else row['home_team']
        total_passes = passes_euro[(passes_euro['match_id'] == match_id) & (passes_euro['team'] == selected_team)].shape[0]
        st.markdown(f"### {selected_team} vs {opponent_team} — Total Passes: {total_passes}")
        fig = plot_pass_network_plotly(
            match_id=match_id, 
            team_name=selected_team,    
            min_passes=min_passes, 
            theme=theme,
            node_color=GOAL_COLOR,
            edge_color=MISS_COLOR
        )
        if fig:
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    else:
        st.info("Please select a match from the sidebar to view the pass network.")