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

# --- 2. EXACT SIDEBAR CSS (MATCHING IMAGE) ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    /* GLOBAL RESET */
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; color: #0f172a; }
    .stApp { background-color: #f8fafc; } /* Light gray background */
    
    /* SIDEBAR (Dark Navy) */
    section[data-testid="stSidebar"] { 
        background-color: #020617; /* The dark color from your image */
        border-right: 1px solid #1e293b;
    }
    
    /* Hide Default Streamlit Sidebar Elements if possible to clean up */
    section[data-testid="stSidebar"] .block-container {
        padding-top: 2rem;
    }

    /* LOGO AREA styling */
    .sidebar-logo-container {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 40px;
        padding-left: 10px;
    }
    .logo-icon {
        width: 32px;
        height: 32px;
        background-color: #2563eb; /* Bright Blue */
        border-radius: 6px;
        color: white;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        font-size: 18px;
    }
    .logo-text {
        font-size: 18px;
        font-weight: 600;
        color: #f8fafc;
    }

    /* MENU HEADER ("Menu") */
    .menu-header {
        color: #94a3b8;
        font-size: 12px;
        font-weight: 600;
        text-transform: uppercase;
        margin-bottom: 10px;
        padding-left: 15px;
        letter-spacing: 0.05em;
    }

    /* RADIO BUTTONS AS MENU LINKS */
    /* 1. Hide the circles */
    div[data-testid="stRadio"] div[role="radiogroup"] > label > div:first-child {
        display: none;
    }
    
    /* 2. Style the labels to look like links */
    div[data-testid="stRadio"] label {
        background-color: transparent;
        padding: 10px 15px;
        border-radius: 6px;
        transition: all 0.2s;
        margin-bottom: 4px;
        color: #cbd5e1 !important; /* Light gray text */
        font-weight: 500;
        cursor: pointer;
    }
    
    /* 3. Hover State */
    div[data-testid="stRadio"] label:hover {
        background-color: #1e293b;
        color: white !important;
    }
    
    /* 4. Active State (No background, just brighter text or slight tint) */
    div[data-testid="stRadio"] label[data-checked="true"] {
        color: #60a5fa !important; /* Blue text highlight */
        background-color: rgba(37, 99, 235, 0.1);
        font-weight: 600;
    }

    /* MAIN CONTENT CARDS */
    .main-card { 
        background-color: white; 
        border-radius: 12px; 
        padding: 30px; 
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05); 
        border: 1px solid #e2e8f0; 
        margin-bottom: 20px; 
    }
    
    /* BUTTONS */
    .stButton>button { 
        background-color: #2563eb; 
        color: white; 
        border-radius: 6px; 
        font-weight: 600; 
        border: none; 
        padding: 0.6rem 1.5rem;
        box-shadow: 0 1px 2px rgba(0,0,0,0.1);
    }
    .stButton>button:hover { background-color: #1d4ed8; }
    
    /* BREADCRUMB */
    .breadcrumb { color: #64748b; font-size: 0.9rem; margin-bottom: 5px; }
    
    /* RESULT BOXES */
    .result-box-pass { border-left: 5px solid #22c55e; background-color: #f0fdf4; padding: 15px; border-radius: 6px; border: 1px solid #bbf7d0; }
    .result-box-fail { border-left: 5px solid #ef4444; background-color: #fef2f2; padding: 15px; border-radius: 6px; border: 1px solid #fecaca; }
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

# --- 4. PAGE: DATA INGESTION ---
def render_ingestion():
    st.markdown('<div class="breadcrumb">Workspace / Data Ingestion</div>', unsafe_allow_html=True)
    st.title("Data Ingestion")
    
    if 'p_df' not in st.session_state: st.session_state['p_df'] = None
    if 'u_df' not in st.session_state: st.session_state['u_df'] = None

    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("1. Internal Policies")
        f1 = st.file_uploader("Upload CSV", key="p", type=["csv"])
        if f1: st.session_state['p_df'] = pd.read_csv(f1); st.success(f"Ready: {len(st.session_state['p_df'])} policies")
    
    with c2:
        st.subheader("2. Regulatory Controls")
        f2 = st.file_uploader("Upload CSV", key="u", type=["csv"])
        if f2: st.session_state['u_df'] = pd.read_csv(f2); st.success(f"Ready: {len(st.session_state['u_df'])} controls")
    st.markdown('</div>', unsafe_allow_html=True)

# --- 5. PAGE: CONTROL MAPPING (AI Action) ---
def render_mapping():
    st.markdown('<div class="breadcrumb">Workspace / Control Mapping</div>', unsafe_allow_html=True)
    st.title("Control Mapping Engine")
    
    if st.session_state.get('p_df') is None or st.session_state.get('u_df') is None:
        st.warning("⚠️ Data missing. Please go to **Data Ingestion** first.")
        return

    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    threshold = st.slider("Match Confidence Threshold", 0, 100, 25)
    st.write("")
    
    if st.button("🚀 Run AI Semantic Mapping"):
        with st.status("Processing Knowledge Base...", expanded=True):
            st.write("Loading Neural Network...")
            embedder = load_embedding_model()
            p_df = st.session_state['p_df']
            u_df = st.session_state['u_df']
            
            p_emb = embedder.encode(p_df['Policy_Text'].head(5).astype(str).tolist(), convert_to_tensor=True)
            u_emb = embedder.encode(u_df['Control_Text'].head(5).astype(str).tolist(), convert_to_tensor=True)
            
            st.write("Generating Audit Plans...")
            bar = st.progress(0)
            
            for i in range(len(p_df.head(5))):
                bar.progress((i+1)/5)
                scores = util.cos_sim(p_emb[i], u_emb)[0]
                best_idx = torch.topk(scores, k=1)[1][0].item()
                ctrl = u_df.iloc[best_idx]
                plan = call_gemini(f"Role: Auditor. Control: {ctrl['Control_Text']}. Policy: {p_df.iloc[i]['Policy_Text']}. Task: Write concise test procedure.")
                db_save_mapping(ctrl['Control_ID'], ctrl['Control_Text'], p_df.iloc[i]['Policy_Text'], plan)
            
            st.success("Mapping Complete & Saved.")
    st.markdown('</div>', unsafe_allow_html=True)

    # Show Mapped Data Table
    df = db_get_mappings()
    if not df.empty:
        st.markdown("### Mapped Controls")
        st.dataframe(df[['control_id', 'control_text', 'status']], use_container_width=True)

# --- 6. PAGE: EVALUATION GUIDANCE (View Only) ---
def render_evaluation():
    st.markdown('<div class="breadcrumb">Workspace / Evaluation Guidance</div>', unsafe_allow_html=True)
    st.title("Evaluation Guidance")
    
    df = db_get_mappings()
    if df.empty: st.info("No mappings found."); return

    st.markdown(f"**Total Controls:** {len(df)}")
    
    for i, row in df.iterrows():
        status_icon = "🟢" if row['status'] == 'PASS' else "🔴" if row['status'] == 'FAIL' else "⚪"
        with st.expander(f"{status_icon} {row['control_id']} (Status: {row['status']})"):
            st.markdown(f"**Control:** {row['control_text']}")
            st.markdown(f"**Policy:** {row['policy_text']}")
            st.divider()
            st.markdown("**🤖 AI Test Plan:**")
            st.info(row['ai_plan'])

# --- 7. PAGE: AUDIT AGENTS (Testing) ---
def render_agents():
    st.markdown('<div class="breadcrumb">Workspace / Audit Agents</div>', unsafe_allow_html=True)
    st.title("Audit Agents")
    
    df = db_get_mappings()
    if df.empty: st.warning("Please map controls first."); return
    
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    c1, c2 = st.columns([1, 2])
    
    with c1:
        selected_id = st.selectbox("Select Control to Audit", df['control_id'])
        record = df[df['control_id'] == selected_id].iloc[0]
        st.info(f"**Req:** {record['control_text']}")

    with c2:
        st.subheader("📡 Evidence Connector")
        f = st.file_uploader("Upload Wiz/AWS JSON", type=['json'], key=f"up_{selected_id}")
        
        if f:
            evidence = json.load(f)
            if st.button("Run AI Audit"):
                with st.spinner("Analyzing..."):
                    prompt = f"Role: Auditor. Control: {record['control_text']}. Evidence: {json.dumps(evidence)}. Output: 'PASS' or 'FAIL' followed by 1 sentence reason."
                    res = call_gemini(prompt)
                    status = "PASS" if "PASS" in res.upper() else "FAIL"
                    db_update_audit(selected_id, status, res, f.name)
                    st.rerun()
    
    if record['status'] != 'Untested':
        color = "result-box-pass" if record['status'] == 'PASS' else "result-box-fail"
        icon = "✅" if record['status'] == 'PASS' else "❌"
        st.markdown(f"""<div class="{color}"><h4>{icon} Latest: {record['status']}</h4><p>{record['last_result']}</p></div>""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown(f"### 📜 Audit History for {selected_id}")
    history = db_get_history(selected_id)
    if not history.empty: 
        st.dataframe(history, use_container_width=True, column_config={"timestamp": "Time", "evidence_source": "Source", "result": "Result", "reason": "AI Analysis"})

# --- 8. PAGE: DASHBOARD (At the End) ---
def render_dashboard():
    st.markdown('<div class="breadcrumb">Workspace / Dashboard</div>', unsafe_allow_html=True)
    st.title("Executive Dashboard")
    
    df = db_get_mappings()
    if df.empty: st.info("No data yet."); return

    total = len(df)
    passed = len(df[df['status'] == 'PASS'])
    failed = len(df[df['status'] == 'FAIL'])
    untested = len(df[df['status'] == 'Untested'])
    score = int((passed / total) * 100) if total > 0 else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Compliance Score", f"{score}%")
    c2.metric("Controls Passed", passed)
    c3.metric("Critical Failures", failed)
    c4.metric("Pending Audit", untested)

    st.write("---")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Control Status")
        fig = px.pie(names=['Passing', 'Failing', 'Untested'], values=[passed, failed, untested], hole=0.5, 
                     color_discrete_sequence=['#22c55e', '#ef4444', '#cbd5e1'])
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.subheader("Risk Velocity")
        domain_data = pd.DataFrame({"Month": ["Jan", "Feb", "Mar", "Apr"], "Issues": [12, 8, 5, 2]})
        st.line_chart(domain_data.set_index("Month"))

# --- 9. SIDEBAR NAVIGATION ---
with st.sidebar:
    # Custom Logo HTML
    st.markdown("""
    <div class="sidebar-logo-container">
        <div class="logo-icon">U</div>
        <div class="logo-text">UCF Platform</div>
    </div>
    <div class="menu-header">Menu</div>
    """, unsafe_allow_html=True)
    
    # Custom Menu Order
    page = st.radio("Nav", [
        "Data Ingestion", 
        "Control Mapping", 
        "Evaluation Guidance", 
        "Audit Agents",
        "Dashboard"
    ], label_visibility="collapsed")
    
    st.write("---")
    st.caption("v3.2 (Production / Gemini)")
    if st.button("Reset DB"):
        c.execute("DELETE FROM mappings"); c.execute("DELETE FROM audit_logs"); conn.commit()
        st.rerun()

# --- 10. ROUTING ---
if page == "Data Ingestion": render_ingestion()
elif page == "Control Mapping": render_mapping()
elif page == "Evaluation Guidance": render_evaluation()
elif page == "Audit Agents": render_agents()
elif page == "Dashboard": render_dashboard()
