import math
import numpy as np
import plotly.graph_objects as go
from typing import Dict, Any, List
from src.io.loaders import load_pathway_network
from src.schemas import PathwayNetwork, ExplanationItem

def annotate_network_with_patient_drivers(
    network: PathwayNetwork, 
    top_drivers: List[ExplanationItem]
) -> PathwayNetwork:
    """Toggle is_driver=True for nodes that are active driver genes in the current patient."""
    driver_symbols = {item.gene_symbol for item in top_drivers if abs(item.contribution) > 0.05}
    
    # Update nodes
    for node in network.nodes:
        if node.id in driver_symbols:
            node.is_driver = True
            
    return network

def render_network_plotly(
    network: PathwayNetwork, 
    explanation_items: List[ExplanationItem]
) -> go.Figure:
    """Render a stunning 2D circular node-link graph using Plotly representing pathway regulatory context."""
    # Build lookup for patient-level details
    patient_lookup = {item.gene_symbol: item for item in explanation_items}
    
    nodes = network.nodes
    edges = network.edges
    
    # Calculate circular layout positions
    # Central nodes (pathways) go to center (0, 0)
    # Surrounding nodes go in a ring at radius 2.0
    pos = {}
    pathway_nodes = [n for n in nodes if n.type == "pathway"]
    other_nodes = [n for n in nodes if n.type != "pathway"]
    
    # Position pathways
    for idx, pw in enumerate(pathway_nodes):
        if len(pathway_nodes) == 1:
            pos[pw.id] = (0.0, 0.0)
        else:
            angle = 2 * math.pi * idx / len(pathway_nodes)
            pos[pw.id] = (0.5 * math.cos(angle), 0.5 * math.sin(angle))
            
    # Position surrounding gene/regulator nodes in a circle
    num_others = len(other_nodes)
    for idx, node in enumerate(other_nodes):
        angle = 2 * math.pi * idx / num_others if num_others else 0.0
        pos[node.id] = (2.0 * math.cos(angle), 2.0 * math.sin(angle))
        
    # Create edge traces
    edge_x = []
    edge_y = []
    
    for edge in edges:
        if edge.source in pos and edge.target in pos:
            x0, y0 = pos[edge.source]
            x1, y1 = pos[edge.target]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1.5, color='#3a3a42'), # elegant dark slate gray line
        hoverinfo='none',
        mode='lines'
    )
    
    # Create node traces
    node_x = []
    node_y = []
    node_text = []
    node_color = []
    node_size = []
    node_line_color = []
    node_line_width = []
    
    for node in nodes:
        if node.id not in pos:
            continue
        x, y = pos[node.id]
        node_x.append(x)
        node_y.append(y)
        
        # Build text hover tooltips
        if node.type == "pathway":
            node_text.append(f"<b>Pathway:</b> {node.label}")
            node_color.append("#3a3a42") # matches legend pathway node
            node_size.append(40)
            node_line_color.append("#e8e8ea")
            node_line_width.append(2)
        else:
            p_info = patient_lookup.get(node.id)
            if p_info:
                # Color node based on contribution: red if increases risk, green if reduces risk
                if p_info.contribution > 0.05:
                    node_color.append("#ef4444") # soft red
                elif p_info.contribution < -0.05:
                    node_color.append("#22c55e") # soft green
                else:
                    node_color.append("#6b6b72") # neutral grey
                    
                role_label = "Transcription Factor" if node.type == "regulator" else "Target Gene"
                tooltip = (
                    f"<b>Gene:</b> {node.id}<br>"
                    f"<b>Type:</b> {role_label}<br>"
                    f"<b>Expression:</b> {p_info.expression_value:.2f} (median: {p_info.cohort_median:.2f})<br>"
                    f"<b>Prognosis:</b> {p_info.direction.capitalize()}<br>"
                    f"<b>Contribution:</b> {p_info.contribution:+.3f}<br>"
                    f"<b>Risk Impact:</b> {p_info.impact.upper()}"
                )
                node_text.append(tooltip)
            else:
                node_color.append("#6b6b72")
                node_text.append(f"<b>Gene:</b> {node.label}<br><i>No patient expression data</i>")
                
            node_size.append(25)
            
            # Highlight patient-specific driver genes with a thick teal border
            if node.is_driver:
                node_line_color.append("#00b8a0")
                node_line_width.append(4)
            else:
                node_line_color.append("#0a0a0b") # matches dark background
                node_line_width.append(1.5)
                
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=[n.id if n.type != "pathway" else "" for n in nodes],
        textposition="top center",
        textfont=dict(color='#e8e8ea', size=11, family='Satoshi, sans-serif'),
        marker=dict(
            showscale=False,
            color=node_color,
            size=node_size,
            line=dict(color=node_line_color, width=node_line_width)
        ),
        hovertext=node_text
    )
    
    # Build figure
    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=20, r=20, t=20),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family='Satoshi, sans-serif', color='#e8e8ea'),
            width=650,
            height=500
        )
    )
    
    return fig
