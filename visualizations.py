import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import networkx as nx
import plotly.colors as pcolors
import matplotlib.pyplot as plt
from data_loader import prepare_shot_map_data, prepare_pass_network_data, calculate_minutes_played, get_player_display_name

def plot_player_radar_chart(player_stats, player_name, theme="green", player_2_stats=None, player_2_name=None):
    """
    Create a radar chart for player statistics using Plotly
    Supports comparison mode with two players
    """
    if not player_stats:
        return None
    
    # Define radar categories and their corresponding normalized stats
    categories = [
        ('Defending', 'defending'),
        ('Passing', 'passing'),
        ('Ball Carrying', 'ball_carrying'),
        ('Dribbling', 'dribbling'),
        ('Creativity', 'creativity'),
        ('Work Rate', 'work_rate'),
        ('Goal Threat', 'goal_threat')
    ]
    
    # Extract values for radar chart
    category_names = [cat[0] for cat in categories]
    values = [player_stats.get(cat[1], 0) for cat in categories]
    
    # Close the radar chart by adding first value at the end
    values += values[:1]
    category_names += category_names[:1]
    
    # Enhanced professional color palettes with refined contrast and visual appeal
    if theme == "white":
        bg_color = "#ffffff"
        line_color = "#2c3e50"
        grid_color = "rgba(189, 195, 199, 0.5)"  # Subtle gray with transparency
        fill_color_1 = "rgba(52, 152, 219, 0.1)"  # Refined blue with subtle transparency
        fill_color_2 = "rgba(231, 76, 60, 0.1)"   # Refined red with subtle transparency
        line_color_1 = "#3498db"  # Brighter blue
        line_color_2 = "#e74c3c"  # Brighter red
        text_color = "#2c3e50"    # Dark blue-gray
        title_color = "#34495e"   # Slightly darker blue-gray
        hover_bg = "#2c3e50"
        hover_text = "#ffffff"
    elif theme == "black":
        bg_color = "#0d1117"      # GitHub Dark theme inspired
        line_color = "#f0f6fc"    # Light gray-blue
        grid_color = "rgba(56, 66, 78, 0.5)"  # Subtle grid lines
        fill_color_1 = "rgba(88, 166, 255, 0.12)"  # GitHub blue with subtle transparency
        fill_color_2 = "rgba(246, 185, 59, 0.12)"  # Gold with subtle transparency
        line_color_1 = "#58a6ff"  # GitHub blue
        line_color_2 = "#f6b93b"  # Gold
        text_color = "#f0f6fc"    # Light gray-blue
        title_color = "#ffffff"
        hover_bg = "#ffffff"
        hover_text = "#000000"
    else:  # green
        bg_color = "#042712"      # Deeper forest green
        line_color = "#ffffff"
        grid_color = "rgba(255, 255, 255, 0.15)"
        fill_color_1 = "rgba(233, 236, 239, 0.10)"  # Light gray with subtle transparency
        fill_color_2 = "rgba(255, 221, 87, 0.10)"   # Gold with subtle transparency
        line_color_1 = "#e9ecef"  # Light gray
        line_color_2 = "#ffd757"  # Gold
        text_color = "#ffffff"
        title_color = "#ffffff"
        hover_bg = "#ffffff"
        hover_text = "#000000"
    
    fig = go.Figure()
    
    # Add first player trace with smoother lines and better styling
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=category_names,
        fill='toself',
        fillcolor=fill_color_1,
        line=dict(color=line_color_1, width=3, smoothing=1.3),
        marker=dict(size=10, color=line_color_1, symbol="circle", line=dict(color="#ffffff", width=1)),
        name=player_name,
        hovertemplate=f'<b>{player_name}</b><br><b>%{{theta}}</b><br>Score: %{{r:.1f}}/100<extra></extra>',
        hoverlabel=dict(
            bgcolor=hover_bg,
            font=dict(color=hover_text, size=14, family="Arial"),
            bordercolor=hover_bg
        ),
        hoverinfo='r+theta'
    ))
    
    # Add second player trace if in comparison mode with smoother lines
    if player_2_stats and player_2_name:
        values_2 = [player_2_stats.get(cat[1], 0) for cat in categories]
        values_2 += values_2[:1]  # Close the chart
        
        fig.add_trace(go.Scatterpolar(
            r=values_2,
            theta=category_names,
            fill='toself',
            fillcolor=fill_color_2,
            line=dict(color=line_color_2, width=3, smoothing=1.3),
            marker=dict(size=10, color=line_color_2, symbol="circle", line=dict(color="#ffffff", width=1)),
            name=player_2_name,
            hovertemplate=f'<b>{player_2_name}</b><br><b>%{{theta}}</b><br>Score: %{{r:.1f}}/100<extra></extra>',
            hoverlabel=dict(
                bgcolor=hover_bg,
                font=dict(color=hover_text, size=14, family="Arial"),
                bordercolor=hover_bg
            ),
            hoverinfo='r+theta'
        ))
    
    # Enhanced layout with more professional styling - FIXED POLAR AXIS CONFIG
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                gridcolor=grid_color,
                gridwidth=1.5,  # Slightly thicker grid lines for better visibility
                tickcolor=text_color,
                tickfont=dict(color=text_color, size=11, family="Arial"),
                tickvals=[20, 40, 60, 80, 100],
                ticktext=["20", "40", "60", "80", "100"],
                linecolor=grid_color,
                linewidth=1,
                showticklabels=True,
                tickangle=0,
                layer="below traces"
            ),
            angularaxis=dict(
                gridcolor=grid_color,
                gridwidth=1.5,  # Slightly thicker grid lines
                tickcolor=text_color,
                tickfont=dict(color=text_color, size=13, family="Arial", weight="bold"),
                linecolor="rgba(0,0,0,0)",  # Hide the circular line
                linewidth=0,
                rotation=90,  # Start from top
                direction="clockwise",
                layer="below traces"
            ),
            bgcolor=bg_color
        ),
        showlegend=bool(player_2_stats and player_2_name),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.15,
            xanchor="center",
            x=0.5,
            font=dict(color=text_color, size=13, family="Arial"),
            bgcolor="rgba(0,0,0,0)",
            bordercolor="rgba(0,0,0,0)"
        ),
        title=dict(
            text=f"<b>{player_name} vs {player_2_name}</b>" if (player_2_stats and player_2_name) else f"<b>{player_name}</b>",
            x=0.5,
            y=0.95,
            font=dict(size=22, color=title_color, family="Arial", weight="bold"),
            xanchor="center"
        ),
        plot_bgcolor=bg_color,
        paper_bgcolor=bg_color,
        height=700,
        width=700,
        margin=dict(l=80, r=80, t=100, b=80),
        font=dict(family="Arial")
    )
    
    # Add performance indicators with improved styling
    if player_stats:
        # Calculate overall performance score
        avg_score = sum(player_stats.get(cat[1], 0) for cat in categories) / len(categories)
        
        # Performance level text with refined thresholds
        if avg_score >= 85:
            performance_text = "Elite"
            performance_color = "#27ae60"
        elif avg_score >= 70:
            performance_text = "Excellent"
            performance_color = "#2ecc71"
        elif avg_score >= 55:
            performance_text = "Good"
            performance_color = "#f39c12"
        elif avg_score >= 40:
            performance_text = "Average"
            performance_color = "#e67e22"
        else:
            performance_text = "Below Average"
            performance_color = "#e74c3c"
        
        # Add performance indicator with better styling
        if not (player_2_stats and player_2_name):  # Only show for single player view
            fig.add_annotation(
                x=0.02, y=0.98,
                xref="paper", yref="paper",
                text=f"<b>Overall: {avg_score:.0f}/100</b><br><span style='color:{performance_color};font-weight:bold'>{performance_text}</span>",
                showarrow=False,
                font=dict(size=14, color=text_color, family="Arial"),
                bgcolor="rgba(255,255,255,0.07)" if theme != "white" else "rgba(0,0,0,0.03)",
                bordercolor=grid_color,
                borderwidth=1,
                borderpad=8,
                align="left",
                xanchor="left",
                yanchor="top"
            )
    
    # Add improved legend explaining what the scores mean
    fig.add_annotation(
        x=-0.12, y=-0.16,
        xref="paper", yref="paper",
        text="<b>Score Explanation:</b><br>0-100 = Percentile rank vs all players<br><i>e.g., 75 = better than 75% of players</i>",
        showarrow=False,
        font=dict(size=12, color=text_color, family="Arial"),
        bgcolor="rgba(255,255,255,0.07)" if theme != "white" else "rgba(0,0,0,0.03)",
        bordercolor=grid_color,
        borderwidth=1,
        borderpad=8,
        align="left",
        xanchor="left",
        yanchor="bottom"
    )
    
    return fig

def plotly_penalty_map_center_only(
    shots_df: pd.DataFrame,
    *,
    outcome_col: str = 'shot_outcome',
    jitter: float = 0.03,
    palette: str = 'neutral',
    theme: str = 'green',
    title: str = '',
    shot_type_label: str = 'Shots',
    GOAL_COLOR: str = "#1f77b4",
    MISS_COLOR: str = "#ff7f0e"
):
    # Prepare shot data
    shot_data = prepare_shot_map_data(shots_df, jitter)
    
    shots = shot_data['shots']
    
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
    is_penalty = shot_data['is_penalty']
    if is_penalty:
        hover_text = shots.apply(lambda row: f"{get_player_display_name(row['player'])}<br>{row['team']} vs {row['opponent_team']}<br>{'Shootout' if row['period']==5 else 'Regular Play'}<br>{row['shot_body_part']}<br>xG: {row.get('shot_statsbomb_xg', 0):.2f}", axis=1)
    else:
        def get_pattern_or_penalty(row):
            if row.get('shot_type', '') == 'Penalty':
                return 'Penalty'
            return row.get('play_pattern', '')
        hover_text = shots.apply(lambda row: f"{get_player_display_name(row['player'])}<br>{row['team']} vs {row['opponent_team']}<br>{get_pattern_or_penalty(row)}<br>{row['shot_body_part']}<br>xG: {row.get('shot_statsbomb_xg', 0):.2f}", axis=1)
    
    if is_penalty:
        sizes = [12] * len(shots)
    else:
        min_size, max_size = 5, 13
        xg_values = shots['shot_statsbomb_xg'].fillna(0)
        xg_sqrt = np.sqrt(xg_values)
        # Add a small epsilon to avoid division by zero if all xG are the same
        if (xg_sqrt.max() - xg_sqrt.min()) > 0:
            sizes = min_size + (max_size - min_size) * (xg_sqrt - xg_sqrt.min()) / (xg_sqrt.max() - xg_sqrt.min())
        else:
            sizes = [min_size] * len(shots)

    fig.add_trace(go.Scatter(
        x=shots['plot_y'],
        y=shots['plot_z'],
        mode='markers',
        marker=dict(
            size=sizes,
            color=[GOAL_COLOR if g else MISS_COLOR for g in shots['is_goal']],
            line=dict(width=1, color=TEXT),
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
    
    return (fig, shot_data['total'], shot_data['goals'], shot_data['not_goals'], 
            shot_data['goal_rate'], shot_data['miss_types'], shot_data['body_part_counts'], 
            shot_data['is_penalty'], shot_data['shootout_count'], shot_data['regular_count'])

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
    return fig

def plot_pass_network_plotly(
    passes_euro, 
    lineups, 
    euro_matches, 
    match_id, 
    team_name, 
    min_passes, 
    theme, 
    node_color, 
    edge_color, 
    xgchain_data
):
    """
    Generates and plots an interactive pass network using Plotly.
    If match_id is None, creates tournament-wide network for the team.
    """
    # Prepare network data based on whether it's match-specific or tournament-wide
    if match_id is not None:
        # Match-specific network (existing functionality)
        network_data = prepare_pass_network_data(passes_euro, lineups, match_id, team_name, xgchain_data)
    else:
        # Tournament-wide network
        from data_loader import prepare_tournament_pass_network_data
        network_data = prepare_tournament_pass_network_data(passes_euro, lineups, team_name, xgchain_data)
    
    if network_data is None:
        return None, None

    # Extract data from prepared network data
    team_passes = network_data['team_passes']
    starting_11 = network_data['starting_11']
    player_positions = network_data['player_positions']
    pass_counts = network_data['pass_counts']
    team_numbers = network_data['team_numbers']
    team_nicknames = network_data['team_nicknames']
    progressive_passes = network_data['progressive_passes']
    pass_success_under_pressure = network_data['pass_success_under_pressure']
    quality_scores = network_data['quality_scores']
    pass_accuracy = network_data['pass_accuracy']

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
        if np.linalg.norm(vec) == 0:
            return x1, y1, x2, y2
        norm_vec = vec / np.linalg.norm(vec)
        perp_vec = np.array([-norm_vec[1], norm_vec[0]]) * offset_perp
        # Move start/end a bit along the direction to avoid overlapping nodes
        start_offset = np.array([x1, y1]) + norm_vec * offset_along + perp_vec
        end_offset = np.array([x2, y2]) - norm_vec * offset_along + perp_vec
        return start_offset[0], start_offset[1], end_offset[0], end_offset[1]

    # Convert to number-based mappings for graph processing
    player_positions_num = {team_numbers.get(p, p): pos for p, pos in player_positions.items()}
    pass_counts_num = {(team_numbers.get(p1, p1), team_numbers.get(p2, p2)): count for (p1, p2), count in pass_counts.items()}
    
    jersey_to_display_name = {str(j): n for n, j in zip(team_nicknames.values(), team_numbers.values())}
    
    pass_counts_num_filtered = {k: v for k, v in pass_counts_num.items() if v >= min_passes}

    # Create full graph for centrality calculation (independent of min_passes filter)
    G_full = nx.DiGraph()
    for (u, v), weight in pass_counts_num.items():
        G_full.add_edge(u, v, weight=weight)
    
    # Create filtered graph for visualization
    G = nx.DiGraph()
    for (u, v), weight in pass_counts_num_filtered.items():
        G.add_edge(u, v, weight=weight)
    
    degrees = dict(G.degree())
    # Calculate centrality on full graph (not filtered)
    centrality_full = nx.degree_centrality(G_full)
    betweenness_centrality_full = nx.betweenness_centrality(G_full)
    
    # Calculate network density
    network_density = nx.density(G_full)

    fig = draw_full_pitch_plotly(theme=theme, fig_width=800, fig_height=600)
    
    nodes_data = list(player_positions_num.items())
    node_jersey_numbers = [item[0] for item in nodes_data]
    node_positions = [item[1] for item in nodes_data]
    node_x = [p[0] for p in node_positions]
    node_y = [p[1] for p in node_positions]

    # Node sizes based on quality score
    player_names_by_jersey = {str(v): k for k, v in team_numbers.items()}
    
    # Scale quality scores to node sizes
    min_quality = min(quality_scores.values()) if quality_scores else 0
    max_quality = max(quality_scores.values()) if quality_scores else 1
    
    min_node_size, max_node_size = 15, 35
    node_sizes = []
    for j in node_jersey_numbers:
        player_name = player_names_by_jersey.get(j, '')
        if max_quality > min_quality:
            normalized_quality = (quality_scores.get(player_name, 0) - min_quality) / (max_quality - min_quality)
            size = min_node_size + (max_node_size - min_node_size) * normalized_quality
        else:
            size = min_node_size
        node_sizes.append(size)
    
    # Node colors based on xGChain
    if xgchain_data:
        xg_values = [xgchain_data.get(player_names_by_jersey.get(j, ''), 0) for j in node_jersey_numbers]
        
        if max(xg_values) > 0:
            norm_xg = [x / max(xg_values) for x in xg_values]
            node_colors = [plt.cm.viridis(x) for x in norm_xg]
            node_colors_rgba = [f'rgba({int(c[0]*255)}, {int(c[1]*255)}, {int(c[2]*255)}, 1)' for c in node_colors]
        else:
            node_colors_rgba = [node_color] * len(node_jersey_numbers)
    else:
        node_colors_rgba = [node_color] * len(node_jersey_numbers)

    # --- Edge Drawing ---
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
            # First direction
            x1_uv, y1_uv, x2_uv, y2_uv = offset_edge_coords(x1, y1, x2, y2, offset_perp=0.8, offset_along=3.0)
            fig.add_annotation(
                x=x2_uv, y=y2_uv, ax=x1_uv, ay=y1_uv,
                xref='x', yref='y', axref='x', ayref='y',
                showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=weight / 4,
                arrowcolor=edge_color, opacity=0.7
            )
            edge_hover_traces.append(go.Scatter(
                x=[x2_uv], y=[y2_uv],
                mode='markers', marker=dict(size=15, color='rgba(0,0,0,0)'),
                hoverinfo='text', hovertext=label_uv, showlegend=False,
                hoverlabel=dict(bgcolor=hover_bg, font=dict(color=hover_font, size=14))
            ))

            # Second direction (return pass)
            weight_vu = pass_counts_num_filtered.get((v, u), 0)
            label_vu = f"{jersey_to_display_name.get(str(v), str(v))} → {jersey_to_display_name.get(str(u), str(u))}: {weight_vu} passes"
            x1_vu, y1_vu, x2_vu, y2_vu = offset_edge_coords(x2, y2, x1, y1, offset_perp=0.8, offset_along=3.0)
            fig.add_annotation(
                x=x2_vu, y=y2_vu, ax=x1_vu, ay=y1_vu,
                xref='x', yref='y', axref='x', ayref='y',
                showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=weight_vu / 4,
                arrowcolor=edge_color, opacity=0.7
            )
            edge_hover_traces.append(go.Scatter(
                x=[x2_vu], y=[y2_vu],
                mode='markers', marker=dict(size=15, color='rgba(0,0,0,0)'),
                hoverinfo='text', hovertext=label_vu, showlegend=False,
                hoverlabel=dict(bgcolor=hover_bg, font=dict(color=hover_font, size=14))
            ))
            
            bidirectional_drawn.add((u, v))
            bidirectional_drawn.add((v, u))
        else: # One-way pass
            x1_offset, y1_offset, x2_offset, y2_offset = offset_edge_coords(x1, y1, x2, y2, offset_perp=0, offset_along=3.0)
            fig.add_annotation(
                x=x2_offset, y=y2_offset, ax=x1_offset, ay=y1_offset,
                xref='x', yref='y', axref='x', ayref='y',
                showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=weight / 4,
                arrowcolor=edge_color, opacity=0.7
            )
            edge_hover_traces.append(go.Scatter(
                x=[x2_offset], y=[y2_offset],
                mode='markers', marker=dict(size=15, color='rgba(0,0,0,0)'),
                hoverinfo='text', hovertext=label_uv, showlegend=False,
                hoverlabel=dict(bgcolor=hover_bg, font=dict(color=hover_font, size=14))
            ))
            bidirectional_drawn.add((u, v))

    # Add invisible edge hover traces
    for trace in edge_hover_traces:
        fig.add_trace(trace)

    fig.update_layout(
        shapes=[dict(layer='below') for shape in fig.layout.shapes]
    )

    # --- Node Drawing ---
    player_pass_counts = team_passes['player'].value_counts().to_dict()
    recipient_pass_counts = team_passes['pass_recipient'].value_counts().to_dict()
    name_to_jersey = {name: jersey for name, jersey in team_numbers.items()}
    passes_by_jersey = {name_to_jersey.get(p, ''): c for p, c in player_pass_counts.items()}
    received_by_jersey = {name_to_jersey.get(p, ''): c for p, c in recipient_pass_counts.items()}

    # Calculate key passes per player
    key_passes_counts = team_passes[team_passes['pass_shot_assist'] == True]['player'].value_counts().to_dict()

    # Calculate centrality ranks
    centrality_ranks = {}
    betweenness_ranks = {}
    if centrality_full:
        sorted_centrality = sorted(centrality_full.items(), key=lambda x: x[1], reverse=True)
        for rank, (player, _) in enumerate(sorted_centrality, 1):
            centrality_ranks[player] = rank
    
    if betweenness_centrality_full:
        sorted_betweenness = sorted(betweenness_centrality_full.items(), key=lambda x: x[1], reverse=True)
        for rank, (player, _) in enumerate(sorted_betweenness, 1):
            betweenness_ranks[player] = rank

    hover_text = []
    for j in node_jersey_numbers:
        player_name = jersey_to_display_name.get(j, 'Unknown')
        full_player_name = player_names_by_jersey.get(j, '')
        total_passes = passes_by_jersey.get(j, 0)
        received_passes = received_by_jersey.get(j, 0)
        centrality_rank = centrality_ranks.get(j, 'N/A')
        betweenness_rank = betweenness_ranks.get(j, 'N/A')
        prog_passes = progressive_passes.get(full_player_name, 0)
        key_passes = key_passes_counts.get(full_player_name, 0)
        pressure_success, pressure_attempts = pass_success_under_pressure.get(full_player_name, (0, 0))
        
        # Calculate minutes played
        minutes_played = calculate_minutes_played(team_passes, full_player_name)
        
        # Get individual pass accuracy
        player_accuracy = pass_accuracy.get(full_player_name, 0)
        
        text = f"#{j} {player_name}<br>Minutes Played: {minutes_played}<br>Passes Given: {total_passes}<br>Passes Received: {received_passes}<br>Pass Accuracy: {player_accuracy:.1f}%<br>Centrality Rank: {centrality_rank}<br>Betweenness Rank: {betweenness_rank}<br>Progressive Passes: {prog_passes}<br>Key Passes: {key_passes}"
        
        if pressure_attempts > 0:
            text += f"<br>Under Pressure: {pressure_success:.1f}% ({pressure_attempts} att)"
        
        if xgchain_data:
            xgc = xgchain_data.get(full_player_name, 0)
            text += f"<br>xGChain: {xgc:.2f}"
        hover_text.append(text)

    fig.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=node_jersey_numbers,
        hovertext=hover_text,
        hoverinfo='text',
        marker=dict(
            color=node_colors_rgba,
            size=[size + 15 for size in node_sizes],
            line=dict(width=2, color='black'),
            opacity=1.0
        ),
        textfont=dict(
            color='white',
            size=14,
            family="Arial, sans-serif"
        ),
        name='Players',
        hoverlabel=dict(bgcolor=hover_bg, font=dict(color=hover_font, size=14))
    ))

    fig.update_layout(showlegend=False)
    
    # Add styled legend box for visual encoding
    legend_x = 3
    legend_y = 77
    legend_width = 34
    legend_height = 12
    
    # Determine colors based on theme - use contrasting colors that pop
    if theme == "white":
        bg_color = "rgba(50, 50, 50, 0.9)"
        border_color = "#333333"
        text_color = "white"
    elif theme == "black":
        bg_color = "rgba(240, 240, 240, 0.95)"
        border_color = "#cccccc"
        text_color = "black"
    else:  # green
        bg_color = "rgba(255, 255, 255, 0.92)"
        border_color = "#2d5e2e"
        text_color = "#2d5e2e"
    
    # Add legend background box
    fig.add_shape(
        type="rect",
        x0=legend_x - 1, y0=legend_y - legend_height + 1,
        x1=legend_x + legend_width, y1=legend_y + 1,
        fillcolor=bg_color,
        line=dict(color=border_color, width=1),
        layer="above"
    )
    
    # Node size explanation with icon - adjusted spacing
    fig.add_annotation(
        x=legend_x + 1, y=legend_y - 0.5,
        text="<span style='font-size:14px'>●</span> <b>Node size</b> = Player Quality",
        showarrow=False,
        font=dict(size=11, color=text_color, family="Arial"),
        xanchor='left', yanchor='top'
    )
    
    fig.add_annotation(
        x=legend_x + 3, y=legend_y - 2.5,
        text="(Pass accuracy + Progressive passes + xGChain)",
        showarrow=False,
        font=dict(size=9, color=text_color, family="Arial"),
        xanchor='left', yanchor='top'
    )
    
    # Edge width explanation with arrow - adjusted spacing
    fig.add_annotation(
        x=legend_x + 1, y=legend_y - 4.5,
        text="<span style='font-size:14px'>→</span> <b>Arrow width</b> = Pass frequency",
        showarrow=False,
        font=dict(size=11, color=text_color, family="Arial"),
        xanchor='left', yanchor='top'
    )
    
    # Node color explanation with color indicator - adjusted spacing
    fig.add_annotation(
        x=legend_x + 1, y=legend_y - 7,
        text="<span style='color:#440154; font-size:14px'>●</span><span style='color:#21918c; font-size:14px'>●</span><span style='color:#fde725; font-size:14px'>●</span> <b>Node color</b> = xGChain value",
        showarrow=False,
        font=dict(size=11, color=text_color, family="Arial"),
        xanchor='left', yanchor='top'
    )
    
    # --- Statistics Calculation ---
    stats = {}

    # Centrality
    if centrality_full:
        max_centrality_node = max(centrality_full, key=centrality_full.get)
        min_centrality_node = min(centrality_full, key=centrality_full.get)
        max_player_name = jersey_to_display_name.get(str(max_centrality_node), str(max_centrality_node))
        min_player_name = jersey_to_display_name.get(str(min_centrality_node), str(min_centrality_node))
        stats['max_centrality'] = (max_player_name, 1, str(max_centrality_node))
        stats['min_centrality'] = (min_player_name, len(centrality_full), str(min_centrality_node))

    # Betweenness Centrality (playmaker identification)
    if betweenness_centrality_full:
        max_betweenness_node = max(betweenness_centrality_full, key=betweenness_centrality_full.get)
        max_betweenness_player = jersey_to_display_name.get(str(max_betweenness_node), str(max_betweenness_node))
        stats['max_betweenness'] = (max_betweenness_player, 1, str(max_betweenness_node))

    # Progressive Passes
    if progressive_passes:
        most_progressive_player = max(progressive_passes.keys(), 
                                    key=lambda x: (progressive_passes[x], player_pass_counts.get(x, 0)))
        if most_progressive_player in team_nicknames:
            most_progressive_display = team_nicknames[most_progressive_player]
        else:
            most_progressive_display = most_progressive_player
        jersey = team_numbers.get(most_progressive_player, '')
        stats['most_progressive'] = (most_progressive_display, progressive_passes[most_progressive_player], jersey)

    # Pass success under pressure
    pressure_players = {p: data for p, data in pass_success_under_pressure.items() if data[1] >= 5}
    if pressure_players:
        best_under_pressure_player = max(pressure_players.keys(), 
                                       key=lambda x: (pressure_players[x][0], pressure_players[x][1]))
        if best_under_pressure_player in team_nicknames:
            best_pressure_display = team_nicknames[best_under_pressure_player]
        else:
            best_pressure_display = best_under_pressure_player
        pressure_rate, pressure_attempts = pressure_players[best_under_pressure_player]
        jersey = team_numbers.get(best_under_pressure_player, '')
        stats['best_under_pressure'] = (best_pressure_display, pressure_rate, pressure_attempts, jersey)

    # Network density
    stats['network_density'] = network_density * 100

    # Most passes given
    if player_pass_counts:
        most_passes_player_name = max(player_pass_counts, key=player_pass_counts.get)
        if most_passes_player_name in team_nicknames:
            most_passes_player_display_name = team_nicknames[most_passes_player_name]
        else:
            most_passes_player_display_name = most_passes_player_name
        jersey = team_numbers.get(most_passes_player_name, '')
        stats['most_passes'] = (most_passes_player_display_name, player_pass_counts[most_passes_player_name], jersey)

    # Most received passes
    recipient_counts = team_passes['pass_recipient'].value_counts()
    if not recipient_counts.empty:
        most_received_player_name = recipient_counts.index[0]
        if most_received_player_name in team_nicknames:
             most_received_player_display_name = team_nicknames[most_received_player_name]
        else:
            most_received_player_display_name = most_received_player_name
        jersey = team_numbers.get(most_received_player_name, '')
        stats['most_received'] = (most_received_player_display_name, recipient_counts.iloc[0], jersey)

    # Pass accuracy - using corrected calculation
    passer_counts = team_passes['player'].value_counts()
    passer_counts = passer_counts[passer_counts >= 30]  # At least 30 passes
    
    if not passer_counts.empty:
        accuracy_dict = {}
        for player in passer_counts.index:
            accuracy_dict[player] = pass_accuracy.get(player, 0)
        
        if accuracy_dict:
            most_accurate_player_name = max(accuracy_dict, key=accuracy_dict.get)
            if most_accurate_player_name in team_nicknames:
                most_accurate_player_display_name = team_nicknames[most_accurate_player_name]
            else:
                most_accurate_player_display_name = most_accurate_player_name
            jersey = team_numbers.get(most_accurate_player_name, '')
            stats['most_accurate'] = (most_accurate_player_display_name, accuracy_dict[most_accurate_player_name], jersey)

    # Most key passes
    key_passes_df = team_passes[team_passes['pass_shot_assist'] == True]
    if not key_passes_df.empty:
        key_passes_counts = key_passes_df['player'].value_counts()
        if not key_passes_counts.empty:
            most_key_passes_player_name = key_passes_counts.index[0]
            if most_key_passes_player_name in team_nicknames:
                most_key_passes_display_name = team_nicknames[most_key_passes_player_name]
            else:
                most_key_passes_display_name = most_key_passes_player_name
            jersey = team_numbers.get(most_key_passes_player_name, '')
            stats['most_key_passes'] = (most_key_passes_display_name, key_passes_counts.iloc[0], jersey)

    return fig, stats