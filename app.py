import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch
import google.generativeai as genai
import json
import sqlite3
import datetime
import plotly.express as px

# --- 1. CONFIG & DB SETUP ---
st.set_page_config(
    page_title="UCF Enterprise Platform",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize SQLite Database
conn = sqlite3.connect('grc_saas.db', check_same_thread=False)
c = conn.cursor()

# Create Tables
c.execute('''CREATE TABLE IF NOT EXISTS mappings 
             (control_id TEXT PRIMARY KEY, control_text TEXT, policy_text TEXT, ai_plan TEXT, status TEXT, last_result TEXT, last_updated TEXT)''')
c.execute('''CREATE TABLE IF NOT EXISTS audit_logs 
             (timestamp TEXT, control_id TEXT, evidence_source TEXT, result TEXT, reason TEXT)''')
conn.commit()

# --- 2. SAAS UI CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; color: #0f172a; }
    .stApp { background-color: #f8fafc; }
    
    /* --- SIDEBAR STYLING --- */
    section[data-testid="stSidebar"] { 
        background-color: #020617; /* Dark Navy */
        border-right: 1px solid #1e293b;
    }
    
    /* LOGO */
    .sidebar-logo {
        color: white !important;
        font-size: 1.5rem;
        font-weight: 700;
        margin-bottom: 30px;
        display: flex;
        align-items: center;
        gap: 10px;
    }

    /* MENU TEXT COLOR FIX (Force White) */
    section[data-testid="stSidebar"] .stRadio label {
        color: #ffffff !important;
        font-weight: 500;
        padding-left: 10px;
    }
    section[data-testid="stSidebar"] .stRadio label p {
        color: #ffffff !important;
        font-size: 15px;
    }
    section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 {
        color: #ffffff !important;
    }
    
    /* Menu Hover & Active States */
    div[data-testid="stRadio"] label:hover {
        background-color: #1e293b;
    }
    div[data-testid="stRadio"] label[data-checked="true"] {
        background-color: #2563eb !important;
        color: white !important;
    }
    div[data-testid="stRadio"] div[role="radiogroup"] > label > div:first-child {
        display: none;
    }

    /* --- DASHBOARD CARDS --- */
    .metric-card {
        background-color: white;
        border-radius: 12px;
        padding: 25px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        border: 1px solid #e2e8f0;
        text-align: left;
        height: 140px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    .metric-value {
        font-size: 2.8rem;
        font-weight: 800;
        margin: 0;
        line-height: 1.2;
    }
    .metric-label {
        color: #64748b;
        font-size: 0.8rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 5px;
    }

    /* GENERAL UI ELEMENTS */
    .main-card { 
        background-color: white; 
        border-radius: 12px; 
        padding: 25px; 
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05); 
        border: 1px solid #e2e8f0; 
        margin-bottom: 20px; 
    }
    
    .stButton>button { 
        background-color: #2563eb; 
        color: white; 
        border-radius: 8px; 
        font-weight: 600; 
        border: none; 
        padding: 0.5rem 1rem; 
    }
    .stButton>button:hover { background-color: #1d4ed8; }
    
    /* Result Boxes */
    .result-box-pass { border-left: 5px solid #22c55e; background-color: #f0fdf4; padding: 15px; border-radius: 6px; }
    .result-box-fail { border-left: 5px solid #ef4444; background-color: #fef2f2; padding: 15px; border-radius: 6px; }
</style>
""", unsafe_allow_html=True)

# --- 3. BACKEND FUNCTIONS ---
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

def call_gemini(prompt):
    try:
        if "GOOGLE_API_KEY" in st.secrets:
            genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
            models = ['gemini-flash-latest', 'gemini-1.5-flash', 'gemini-pro']
            for m in models:
                try:
                    model = genai.GenerativeModel(m)
                    return model.generate_content(prompt).text
                except: continue
            return "❌ Error: API unavailable."
        return "Error: API Key missing."
    except Exception as e: return f"Error: {e}"

def db_save_mapping(cid, ctext, ptext, plan):
    c.execute("INSERT OR REPLACE INTO mappings (control_id, control_text, policy_text, ai_plan, status, last_result, last_updated) VALUES (?, ?, ?, ?, ?, ?, ?)",
              (cid, ctext, ptext, plan, 'Untested', None, None))
    conn.commit()

def db_update_audit(cid, status, result_text, filename):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute("UPDATE mappings SET status = ?, last_result = ?, last_updated = ? WHERE control_id = ?",
              (status, result_text, ts, cid))
    c.execute("INSERT INTO audit_logs VALUES (?, ?, ?, ?, ?)", (ts, cid, filename, status, result_text))
    conn.commit()

def db_get_mappings():
    return pd.read_sql("SELECT * FROM mappings", conn)

def db_get_control(cid):
    c.execute("SELECT * FROM mappings WHERE control_id = ?", (cid,))
    return c.fetchone()

def db_get_history(cid):
    return pd.read_sql("SELECT timestamp, evidence_source, result, reason FROM audit_logs WHERE control_id = ? ORDER BY timestamp DESC", conn, params=(cid,))

# --- 4. PAGE: INGESTION ---
def render_ingestion():
    st.markdown("### 📂 Data Ingestion")
    
    if 'p_df' not in st.session_state: st.session_state['p_df'] = None
    if 'u_df' not in st.session_state: st.session_state['u_df'] = None

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="main-card"><h3>1. Policies</h3>', unsafe_allow_html=True)
        f1 = st.file_uploader("Internal Policies CSV", key="p", type=["csv"])
        if f1: st.session_state['p_df'] = pd.read_csv(f1); st.success(f"Loaded {len(st.session_state['p_df'])} rows")
        st.markdown('</div>', unsafe_allow_html=True)
        
    with c2:
        st.markdown('<div class="main-card"><h3>2. Controls</h3>', unsafe_allow_html=True)
        f2 = st.file_uploader("Regulatory Controls CSV", key="u", type=["csv"])
        if f2: st.session_state['u_df'] = pd.read_csv(f2); st.success(f"Loaded {len(st.session_state['u_df'])} rows")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("### 🚀 AI Core")
    if st.button("Run AI Semantic Mapping", use_container_width=True):
        if st.session_state['p_df'] is None or st.session_state['u_df'] is None:
            st.error("Please upload both CSV files first.")
            return
            
        with st.status("🧠 AI Mapping & Database Sync...", expanded=True):
            st.write("Vectorizing Data...")
            embedder = load_embedding_model()
            p_df, u_df = st.session_state['p_df'], st.session_state['u_df']
            
            p_emb = embedder.encode(p_df['Policy_Text'].head(5).astype(str).tolist(), convert_to_tensor=True)
            u_emb = embedder.encode(u_df['Control_Text'].head(5).astype(str).tolist(), convert_to_tensor=True)
            
            st.write("Generating Plans & Saving to DB...")
            bar = st.progress(0)
            
            for i in range(len(p_df.head(5))):
                bar.progress((i+1)/5)
                scores = util.cos_sim(p_emb[i], u_emb)[0]
                best_idx = torch.topk(scores, k=1)[1][0].item()
                ctrl = u_df.iloc[best_idx]
                plan = call_gemini(f"Role: Auditor. Control: {ctrl['Control_Text']}. Policy: {p_df.iloc[i]['Policy_Text']}. Task: Write 3-step audit plan.")
                db_save_mapping(ctrl['Control_ID'], ctrl['Control_Text'], p_df.iloc[i]['Policy_Text'], plan)
            
            st.success("✅ Mapping Saved to Database!")

# --- 5. PAGE: MAPPED CONTROLS ---
def render_view_mappings():
    st.markdown("### 🔗 Mapped Controls")
    df = db_get_mappings()
    if df.empty: st.info("Database is empty. Go to **Ingestion**."); return

    st.markdown(f"**Total Controls:** {len(df)}")
    for i, row in df.iterrows():
        icon = "🟢" if row['status'] == 'PASS' else "🔴" if row['status'] == 'FAIL' else "⚪"
        with st.expander(f"{icon} {row['control_id']} (Status: {row['status']})"):
            st.markdown(f"**Control:** {row['control_text']}")
            st.markdown(f"**Policy:** {row['policy_text']}")
            st.caption(f"**Test Plan:** {row['ai_plan']}")

# --- 6. PAGE: CONTROL TESTING ---
def render_testing():
    st.markdown("### 🤖 Automated Control Testing")
    
    df = db_get_mappings()
    if df.empty: st.warning("No mappings found."); return
    
    # 1. SELECT CONTROL
    c1, c2 = st.columns([1, 2])
    with c1:
        st.markdown('<div class="main-card">', unsafe_allow_html=True)
        selected_id = st.selectbox("Select Control", df['control_id'])
        record = db_get_control(selected_id)
        st.info(f"**Requirement:** {record[1]}")
        st.markdown('</div>', unsafe_allow_html=True)

    # 2. UPLOAD & TEST
    with c2:
        st.markdown('<div class="main-card"><h3>📡 Evidence Connector</h3>', unsafe_allow_html=True)
        f = st.file_uploader("Upload Wiz/AWS JSON", type=['json'], key=f"up_{selected_id}")
        
        if f:
            evidence = json.load(f)
            st.json(evidence, expanded=False)
            
            if st.button("Run AI Audit"):
                with st.spinner("Analyzing..."):
                    prompt = f"Role: Auditor. Control: {record[1]}. Evidence: {json.dumps(evidence)}. Output: 'PASS' or 'FAIL' followed by 1 sentence reason."
                    res = call_gemini(prompt)
                    status = "PASS" if "PASS" in res.upper() else "FAIL"
                    
                    # Update DB (Pass filename as source)
                    db_update_audit(selected_id, status, res, f.name)
                    st.rerun()

        # 3. LATEST RESULT
        current_status = record[4]
        current_result = record[5]
        
        if current_status != 'Untested' and current_result:
            css_class = "result-box-pass" if current_status == "PASS" else "result-box-fail"
            icon = "✅" if current_status == "PASS" else "❌"
            st.markdown(f"""<div class="{css_class}"><h4>{icon} Latest: {current_status}</h4><p>{current_result}</p></div>""", unsafe_allow_html=True)
            
        st.markdown('</div>', unsafe_allow_html=True)

    # 4. AUDIT HISTORY
    st.markdown(f"### 📜 Audit History for {selected_id}")
    history_df = db_get_history(selected_id)
    
    if not history_df.empty:
        st.dataframe(
            history_df, 
            use_container_width=True,
            column_config={
                "timestamp": "Time",
                "evidence_source": "Evidence File",
                "result": "Status",
                "reason": "AI Analysis"
            }
        )
    else:
        st.info("No audit history found for this control.")

# --- 7. PAGE: DASHBOARD (MATCHING IMAGE) ---
def render_dashboard():
    # Header Icon and Title
    st.markdown("""
        <h1 style="display: flex; align-items: center; gap: 10px; margin-bottom: 20px;">
            <span style="font-size: 2.5rem;">📊</span> Executive Dashboard
        </h1>
    """, unsafe_allow_html=True)
    
    df = db_get_mappings()
    if df.empty: st.info("No data yet. Run mappings first."); return

    # --- CALCULATION ---
    total = len(df)
    passed = len(df[df['status'] == 'PASS'])
    failed = len(df[df['status'] == 'FAIL'])
    untested = len(df[df['status'] == 'Untested'])
    score = int((passed / total) * 100) if total > 0 else 0

    # --- TOP ROW: 4 METRIC CARDS ---
    c1, c2, c3, c4 = st.columns(4)
    
    # Helper to generate HTML card (Matching Image Style)
    def kpi_card(val, label, color="#0f172a"):
        return f"""
        <div class="metric-card">
            <h1 class="metric-value" style="color: {color}">{val}</h1>
            <p class="metric-label">{label}</p>
        </div>
        """

    c1.markdown(kpi_card(f"{score}%", "COMPLIANCE SCORE", "#0f172a"), unsafe_allow_html=True)
    c2.markdown(kpi_card(passed, "PASSING", "#22c55e"), unsafe_allow_html=True)
    c3.markdown(kpi_card(failed, "FAILING", "#ef4444"), unsafe_allow_html=True)
    c4.markdown(kpi_card(untested, "UNTESTED", "#64748b"), unsafe_allow_html=True)

    st.write("")
    st.write("")

    # --- BOTTOM ROW: 2 CHARTS ---
    col_left, col_right = st.columns(2)
    
    # 1. DONUT CHART (Control Status)
    with col_left:
        st.markdown("### Control Status")
        
        # Data
        status_data = pd.DataFrame({
            'Status': ['Passing', 'Failing', 'Untested'],
            'Count': [passed, failed, untested]
        })
        
        # Donut Config
        fig = px.pie(
            status_data, 
            values='Count', 
            names='Status', 
            hole=0.5, # Donut shape
            color='Status', 
            color_discrete_map={
                'Passing': '#22c55e', 
                'Failing': '#ef4444', 
                'Untested': '#cbd5e1'
            }
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(showlegend=True, margin=dict(t=0, b=0, l=0, r=0))
        
        # Display in Card
        st.markdown('<div class="main-card">', unsafe_allow_html=True)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # 2. BAR CHART (Risk by Domain)
    with col_right:
        st.markdown("### Risk by Domain")
        
        # Creating dynamic mock data for display
        # In a real app, you would have a 'domain' column in your CSV
        domain_data = pd.DataFrame({
            "Domain": ["Access", "Encryption", "Logging", "Network"],
            "Risk": [10, 80, 20, 40]  # Example values
        })
        
        # Bar Config
        fig2 = px.bar(
            domain_data, 
            x="Domain", 
            y="Risk", 
            color="Risk", 
            color_continuous_scale="Reds", # Heatmap style red
            text="Risk"
        )
        fig2.update_layout(
            showlegend=False, 
            xaxis_title="", 
            yaxis_title="Risk",
            coloraxis_showscale=True,
            margin=dict(t=0, b=0, l=0, r=0)
        )
        
        # Display in Card
        st.markdown('<div class="main-card">', unsafe_allow_html=True)
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

# --- NAVIGATION ---
with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
        <div style="background:#2563eb; width:30px; height:30px; border-radius:6px; display:flex; align-items:center; justify-content:center;">U</div>
        UCF Platform
    </div>
    """, unsafe_allow_html=True)
    
    page = st.radio("Menu", ["Ingestion", "Mapped Controls", "Control Testing", "Dashboard"], label_visibility="collapsed")
    
    st.write("---")
    if st.button("Reset Database"):
        c.execute("DELETE FROM mappings"); c.execute("DELETE FROM audit_logs"); conn.commit()
        st.rerun()

# --- ROUTING ---
if page == "Ingestion": render_ingestion()
elif page == "Mapped Controls": render_view_mappings()
elif page == "Control Testing": render_testing()
elif page == "Dashboard": render_dashboard()
