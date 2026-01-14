import streamlit as st

# Icon URL for Favicon
ICON_HEART = "https://cdn-icons-png.flaticon.com/512/2966/2966486.png"

def setup_page():
    st.set_page_config(
        page_title="MedAI Cardiac Risk",
        page_icon=ICON_HEART, 
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.markdown("""
        <style>
        /* --- Palette ---
           Primary: #2F80ED (Blue)
           Secondary: #56CCF2 (Cyan)
           Background: #1F2D3D (Dark Blue)
           Surface: #2C3E50 (Card BG)
           Text: #E0F7FA (Light Cyan)
        */

        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

        /* Global Styles */
        .stApp { 
            background-color: #1F2D3D; 
            color: #E0F7FA; 
            font-family: 'Inter', sans-serif;
        }
        
        /* Sidebar */
        [data-testid="stSidebar"] { 
            background-color: #15202B; 
            border-right: 1px solid rgba(47, 128, 237, 0.2);
        }
        
        /* Main Container */
        .main .block-container {
            max_width: 1200px !important;
            padding-top: 3rem !important;
            padding-bottom: 3rem !important;
        }
        
        /* Typography */
        h1, h2, h3, h4, h5, h6 {
            color: #E0F7FA !important;
            font-weight: 700 !important;
            letter-spacing: -0.5px;
        }
        h1 { font-size: 2.5rem !important; }
        h2 { font-size: 1.8rem !important; color: #56CCF2 !important; }
        h3 { font-size: 1.4rem !important; }
        
        p, label, span, div {
            color: #B0BEC5 !important; /* Softer text color */
            font-weight: 400;
        }
        
        /* Inputs (Modern Flat Style) */
        .stTextInput input, .stSelectbox div[data-baseweb="select"], .stNumberInput input, .stTextArea textarea {
            background-color: rgba(44, 62, 80, 0.5);
            color: #E0F7FA;
            border: 1px solid rgba(86, 204, 242, 0.3);
            border-radius: 8px;
            padding: 0.5rem;
            transition: all 0.2s;
        }
        .stTextInput input:focus, .stTextArea textarea:focus {
            border-color: #2F80ED;
            background-color: rgba(44, 62, 80, 0.8);
            box-shadow: 0 0 0 2px rgba(47, 128, 237, 0.2);
        }
        
        /* Buttons */
        .stButton>button { 
            width: 100%; 
            border-radius: 8px; 
            font-weight: 600; 
            font-size: 14px;
            padding: 0.6rem 1.2rem;
            border: none;
            letter-spacing: 0.5px;
            transition: all 0.2s ease;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        /* Primary Button */
        .stButton>button[kind="primary"] {
            background: linear-gradient(135deg, #2F80ED 0%, #56CCF2 100%);
            color: white !important;
        }
        .stButton>button[kind="primary"]:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(47, 128, 237, 0.4);
        }
        
        /* Secondary Button */
        .stButton>button[kind="secondary"] {
            background-color: rgba(255, 255, 255, 0.05);
            color: #56CCF2 !important;
            border: 1px solid rgba(86, 204, 242, 0.3);
        }
        .stButton>button[kind="secondary"]:hover {
            background-color: rgba(255, 255, 255, 0.1);
            border-color: #56CCF2;
        }
        
        /* Cards & Metrics */
        .metric-container { 
            background-color: rgba(44, 62, 80, 0.6);
            backdrop-filter: blur(10px);
            padding: 20px; 
            border-radius: 12px; 
            border: 1px solid rgba(255, 255, 255, 0.05);
            box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        }
        
        /* Tabs (Modern Pill Style) */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            background-color: rgba(0,0,0,0.2);
            padding: 5px;
            border-radius: 10px;
            border: none;
            margin-bottom: 25px;
            width: fit-content;
        }
        .stTabs [data-baseweb="tab"] {
            height: 40px;
            border-radius: 8px;
            border: none;
            color: #B0BEC5;
            font-weight: 500;
            padding: 0 20px;
            background-color: transparent;
        }
        .stTabs [aria-selected="true"] {
            background-color: #2F80ED !important;
            color: white !important;
            box-shadow: 0 2px 8px rgba(0,0,0,0.2);
        }
        
        /* Form Styling */
        [data-testid="stForm"] {
            background-color: rgba(44, 62, 80, 0.4);
            padding: 2.5rem;
            border-radius: 16px;
            border: 1px solid rgba(255, 255, 255, 0.05);
            box-shadow: 0 8px 32px rgba(0,0,0,0.2);
        }
        
        /* Expander */
        .streamlit-expanderHeader {
            background-color: rgba(255, 255, 255, 0.03);
            border-radius: 8px;
            border: 1px solid rgba(255, 255, 255, 0.05);
        }
        
        /* Divider */
        hr {
            border-color: rgba(255, 255, 255, 0.1);
            margin: 2rem 0;
        }
        
        /* Custom Alert Box */
        .custom-alert {
            padding: 16px;
            border-radius: 8px;
            display: flex;
            align-items: center;
            margin-bottom: 16px;
            backdrop-filter: blur(5px);
        }
        </style>
        """, unsafe_allow_html=True)
