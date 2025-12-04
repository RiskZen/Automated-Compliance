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
    page_title="UCF Platform",
    page_icon="U",
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

# --- 2. EXACT UI STYLING (MATCHING SCREENSHOT) ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    /* GLOBAL RESET */
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; color: #0f172a; }
    .stApp { background-color: #f8fafc; } /* Light gray background */
    
    /* SIDEBAR (Dark Navy) */
    section[data-testid="stSidebar"] { background-color: #020617; border-right: 1px solid #1e293b; }
    section[data-testid="stSidebar"] * { color: #94a3b8 !important; font-weight: 500; }
    
    /* LOGO AREA */
    .sidebar-logo { color: white !important; font-size: 1.4rem; font-weight: 700; margin-bottom: 1.5rem; display: flex; align-items: center; gap: 10px; }
    
    /* MENU ITEMS */
    div[data-testid="stRadio"] label { padding: 10px 15px; border-radius: 6px; transition: all 0.2s; margin-bottom: 2px; }
    div[data-testid="stRadio"] label:hover { background-color: #1e293b; color: white !important; }
    div[data-testid="stRadio"] label[data-checked="true"] { background-color: #2563eb !important; color: white !important; font-weight: 600; }
    
    /* BREADCRUMB */
    .breadcrumb { color: #64748b; font-size: 0.9rem; margin-bottom: 5px; }
    
    /* PAGE TITLES */
    h1 { font-size: 2.2rem; font-weight: 800; color: #0f172a; letter-spacing: -0.03em; margin-bottom: 30px; }
    
    /* MAIN CARD CONTAINER */
    .main-card { 
        background-color: white; 
        border-radius: 12px; 
        padding: 40px; 
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05); 
        border: 1px solid #e2e8f0; 
        margin-bottom: 20px; 
    }
    
    /* BUTTONS (Bright Blue) */
    .stButton>button { 
        background-color: #2563eb; 
        color: white; 
        border-radius: 8px; 
        font-weight: 600; 
        border: none; 
        padding: 0.7rem 1.5rem;
        width: 100%;
    }
    .stButton>button:hover { background-color: #1d4ed8; }
    
    /* STATUS BADGES */
    .status-pass { background-color: #dcfce7; color: #166534; padding: 4px 10px; border-radius: 6px; font-weight: 600; font-size: 12px; }
    .status-fail { background-color: #fee2e2; color: #991b1b; padding: 4px 10px; border-radius: 6px; font-weight: 600; font-size: 12px; }
</style>
""", unsafe_allow_html=True)

# --- 3. BACKEND LOGIC ---
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

def db_update_audit(cid, status, result_text):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute("UPDATE mappings SET status = ?, last_result = ?, last_updated = ? WHERE control_id = ?",
              (status, result_text, ts, cid))
    c.execute("INSERT INTO audit_logs VALUES (?, ?, ?, ?, ?)", (ts, cid, "Manual/Upload", status, result_text))
    conn.commit()

def db_get_mappings():
    return pd.read_sql("SELECT * FROM mappings", conn)

def db_get_history(cid):
    return pd.read_sql("SELECT timestamp, evidence_source, result, reason FROM audit_logs WHERE control_id = ? ORDER BY timestamp DESC", conn, params=(cid,))

# --- 4. PAGE: DASHBOARD ---
def render_dashboard():
    st.markdown('<div class="breadcrumb">Workspace / Dashboard</div>', unsafe_allow_html=True)
    st.title("Executive Dashboard")
    
    df = db_get_mappings()
    if df.empty:
        st.info("No data available. Please go to **Data Ingestion** to start.")
        return

    # Metrics
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

# --- 5. PAGE: DATA INGESTION ---
def render_ingestion():
    st.markdown('<div class="breadcrumb">Workspace / Data Ingestion</div>', unsafe_allow_html=True)
    st.title("Data Ingestion")
    
    # Session state for files (temporary holding before DB)
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

# --- 6. PAGE: CONTROL MAPPING (THE SCREENSHOT MATCH) ---
def render_mapping():
    st.markdown('<div class="breadcrumb">Workspace / Control Mapping</div>', unsafe_allow_html=True)
    st.title("Control Mapping Engine")
    
    # Container for the clean white look
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    
    # Data Check
    if st.session_state.get('p_df') is None or st.session_state.get('u_df') is None:
        st.warning("⚠️ Data missing. Please go to **Data Ingestion** and upload files first.")
        st.markdown('</div>', unsafe_allow_html=True)
        return

    # Slider
    threshold = st.slider("Match Confidence Threshold", 0, 100, 25)
    st.caption("Adjust the sensitivity of the AI semantic matching engine.")
    
    st.write("") # Spacer
    
    # Big Blue Button
    if st.button("🚀 Run AI Semantic Mapping"):
        with st.status("Processing Knowledge Base...", expanded=True):
            st.write("Loading Neural Network...")
            embedder = load_embedding_model()
            p_df = st.session_state['p_df']
            u_df = st.session_state['u_df']
            
            st.write("Vectorizing Content...")
            p_emb = embedder.encode(p_df['Policy_Text'].head(5).astype(str).tolist(), convert_to_tensor=True)
            u_emb = embedder.encode(u_df['Control_Text'].head(5).astype(str).tolist(), convert_to_tensor=True)
            
            st.write("Generating Audit Plans...")
            bar = st.progress(0)
            
            # Logic
            for i in range(len(p_df.head(5))):
                bar.progress((i+1)/5)
                scores = util.cos_sim(p_emb[i], u_emb)[0]
                best_idx = torch.topk(scores, k=1)[1][0].item()
                ctrl = u_df.iloc[best_idx]
                
                # Gemini Call
                plan = call_gemini(f"Role: Auditor. Control: {ctrl['Control_Text']}. Policy: {p_df.iloc[i]['Policy_Text']}. Task: Write concise test procedure.")
                
                # DB Save
                db_save_mapping(ctrl['Control_ID'], ctrl['Control_Text'], p_df.iloc[i]['Policy_Text'], plan)
            
            st.success("Mapping Complete & Saved to Database.")
            
    st.markdown('</div>', unsafe_allow_html=True)

# --- 7. PAGE: EVALUATION GUIDANCE (View Mappings) ---
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

# --- 8. PAGE: AUDIT AGENTS (Testing) ---
def render_agents():
    st.markdown('<div class="breadcrumb">Workspace / Audit Agents</div>', unsafe_allow_html=True)
    st.title("Audit Agents")
    
    df = db_get_mappings()
    if df.empty: st.warning("Please map controls first."); return
    
    # Selection Area
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
                    db_update_audit(selected_id, status, res)
                    st.rerun()
    
    # Current Result Display
    if record['status'] != 'Untested':
        color = "#dcfce7" if record['status'] == 'PASS' else "#fee2e2"
        st.markdown(f"""
        <div style="background-color:{color}; padding:15px; border-radius:8px; margin-top:20px;">
            <strong>Latest Result: {record['status']}</strong><br>{record['last_result']}
        </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # History Table
    st.subheader("📜 Audit History")
    history = db_get_history(selected_id)
    if not history.empty: st.dataframe(history, use_container_width=True)

# --- 9. SIDEBAR NAVIGATION ---
with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
        <div style="background:#2563eb; width:32px; height:32px; border-radius:6px; display:flex; align-items:center; justify-content:center;">U</div>
        UCF Platform
    </div>
    <div style="margin-bottom:20px; color:#64748b; font-size:0.8rem; font-weight:600;">Menu</div>
    """, unsafe_allow_html=True)
    
    # Match Screenshot Order
    page = st.radio("Nav", ["Dashboard", "Data Ingestion", "Control Mapping", "Evaluation Guidance", "Audit Agents"], label_visibility="collapsed")
    
    st.write("---")
    st.caption("v3.0 (Cloud Edition / Gemini)")
    if st.button("Reset DB"):
        c.execute("DELETE FROM mappings"); c.execute("DELETE FROM audit_logs"); conn.commit()
        st.rerun()

# --- 10. ROUTING ---
if page == "Dashboard": render_dashboard()
elif page == "Data Ingestion": render_ingestion()
elif page == "Control Mapping": render_mapping()
elif page == "Evaluation Guidance": render_evaluation()
elif page == "Audit Agents": render_agents()
