import streamlit as st

def apply_custom_css():
    """Apply premium clinical dark-theme styles, Vercel-like typography, and card panels to the application."""
    custom_css = """
    <style>
    /* Import Satoshi Font */
    @import url('https://api.fontshare.com/v2/css?f[]=satoshi@300,400,500,600,700&display=swap');
    
    /* Root variable overrides for Streamlit */
    :root {
        --primary-color: #00b8a0 !important;
        --background-color: #0a0a0b !important;
        --secondary-background-color: #111113 !important;
    }
    
    /* Reset margins and set font globally */
    html, body, [class*="css"], .stApp {
        font-family: 'Satoshi', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
        background-color: #0a0a0b !important;
        color: #e8e8ea !important;
        letter-spacing: -0.01em;
    }
    
    /* Side Bar layout overrides */
    [data-testid="stSidebar"] {
        background-color: #111113 !important;
        border-right: 1px solid rgba(255,255,255,0.07) !important;
        padding-top: 10px;
    }
    [data-testid="stSidebar"] [class*="css"] {
        background-color: #111113 !important;
    }
    
    /* Section Headings - small caps, tracked, hairline divider */
    .section-label {
        font-size: 10px !important;
        font-weight: 700 !important;
        letter-spacing: 0.12em !important;
        text-transform: uppercase !important;
        color: #a1a1aa !important;
        margin-top: 28px !important;
        margin-bottom: 8px !important;
        display: flex;
        align-items: center;
    }
    .section-divider {
        height: 1px;
        background: rgba(255,255,255,0.07) !important;
        margin-bottom: 20px !important;
        margin-top: 4px;
        width: 100%;
    }
    
    /* Clean rectangular card panels */
    .onco-card {
        background: #18181c !important;
        border: 1px solid rgba(255,255,255,0.07) !important;
        border-radius: 4px !important;
        padding: 20px 24px !important;
        margin-bottom: 16px !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4);
        transition: border-color 0.15s ease;
    }
    .onco-card:hover {
        border-color: rgba(255, 255, 255, 0.12) !important;
    }
    
    /* Status indicators with colored left borders (clinical, not pill buttons) */
    .status-indicator {
        padding: 10px 16px;
        border-radius: 4px;
        background-color: #18181c;
        border: 1px solid rgba(255, 255, 255, 0.07);
        font-size: 0.9rem;
        font-weight: 600;
        display: inline-block;
        margin-bottom: 10px;
    }
    .status-low {
        border-left: 4px solid #22c55e !important;
        color: #22c55e;
    }
    .status-intermediate {
        border-left: 4px solid #f59e0b !important;
        color: #f59e0b;
    }
    .status-high {
        border-left: 4px solid #ef4444 !important;
        color: #ef4444;
    }
    
    /* Linear spectrum risk bar */
    .spectrum-container {
        position: relative;
        margin: 20px 0;
        padding-top: 10px;
    }
    .spectrum-bar {
        height: 8px;
        border-radius: 4px;
        background: linear-gradient(to right, #22c55e, #f59e0b, #ef4444);
        width: 100%;
        position: relative;
    }
    .spectrum-pin {
        position: absolute;
        top: -6px;
        width: 4px;
        height: 20px;
        background-color: #ffffff;
        border-radius: 2px;
        box-shadow: 0 0 6px rgba(0, 0, 0, 0.8);
        transform: translateX(-50%);
    }
    .spectrum-labels {
        display: flex;
        justify-content: space-between;
        font-size: 10px;
        color: #a1a1aa;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 6px;
        font-weight: 600;
    }
    
    /* Metric typography */
    .metric-value {
        font-size: 2.2rem !important;
        font-weight: 700 !important;
        color: #ffffff !important;
        letter-spacing: -0.02em;
    }
    .metric-label {
        font-size: 11px !important;
        text-transform: uppercase !important;
        letter-spacing: 0.08em !important;
        color: #a1a1aa !important;
        font-weight: 600 !important;
    }
    
    /* Custom button styles */
    div.stButton > button {
        background-color: #111113 !important;
        border: 1px solid #334155 !important;
        color: #e8e8ea !important;
        font-size: 0.9rem !important;
        font-weight: 600 !important;
        padding: 8px 16px !important;
        border-radius: 4px !important;
        transition: all 0.15s ease !important;
        width: 100%;
    }
    div.stButton > button:hover {
        background-color: #00b8a0 !important;
        border-color: #00b8a0 !important;
        color: #0a0a0b !important;
    }
    
    /* Main primary button override (sidebar run) */
    div[data-testid="stSidebar"] div.stButton > button {
        background-color: #111113 !important;
        border: 1px solid rgba(0, 184, 160, 0.4) !important;
        color: #00b8a0 !important;
    }
    div[data-testid="stSidebar"] div.stButton > button:hover {
        background-color: #00b8a0 !important;
        border-color: #00b8a0 !important;
        color: #0a0a0b !important;
    }
    
    /* Tab headers redesign (plain text caps, clean border active) */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0px !important;
        background-color: #111113 !important;
        padding: 0px !important;
        border-bottom: 1px solid rgba(255,255,255,0.07) !important;
        border-radius: 0px !important;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 12px 24px !important;
        border-radius: 0px !important;
        color: #a1a1aa !important;
        background-color: transparent !important;
        border: none !important;
        font-size: 12px !important;
        font-weight: 700 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.08em !important;
        transition: all 0.15s ease !important;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #e8e8ea !important;
    }
    .stTabs [aria-selected="true"] {
        color: #00b8a0 !important;
        border-bottom: 2px solid #00b8a0 !important;
        background-color: transparent !important;
    }
    
    /* Scientific table overrides */
    .scientific-table {
        width: 100%;
        border-collapse: collapse;
        margin-top: 10px;
        font-size: 13px;
        color: #e8e8ea;
    }
    .scientific-table th {
        background-color: #111113;
        color: #a1a1aa;
        text-align: left;
        padding: 10px 12px;
        border-bottom: 1px solid rgba(255,255,255,0.07);
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-size: 11px;
    }
    .scientific-table td {
        padding: 10px 12px;
        border-bottom: 1px solid rgba(255,255,255,0.04);
    }
    .scientific-table tr:hover {
        background-color: rgba(255, 255, 255, 0.02);
    }
    
    /* Clean layout formatting for metric components */
    div[data-testid="stMetric"], 
    div[data-testid="stColumn"] > div,
    .stTabs [data-baseweb="tab-panel"] {
        border-radius: 4px !important;
    }
    
    /* Radio lists/Segmented controls styling */
    div[data-testid="stRadio"] > div {
        background-color: #18181c;
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 4px;
        padding: 6px;
    }
    
    /* Sidebar dots style */
    .risk-dot {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        margin-right: 6px;
    }
    .dot-low { background-color: #22c55e; }
    .dot-high { background-color: #ef4444; }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)
