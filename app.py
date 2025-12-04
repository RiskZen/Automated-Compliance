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

# --- 2. EXACT SIDEBAR CSS MATCHING IMAGE ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    /* GLOBAL RESET */
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; color: #0f172a; }
    .stApp { background-color: #f8fafc; }
    
    /* --- SIDEBAR CONTAINER --- */
    section[data-testid="stSidebar"] { 
        background-color: #020617; 
        border-right: 1px solid #1e293b;
    }
    
    /* --- MENU STYLING --- */
    .menu-header {
        color: #94a3b8;
        font-size: 13px;
        font-weight: 600;
        margin-top: 30px;
        margin-bottom: 10px;
        padding-left: 10px;
    }
    
    div[data-testid="stRadio"] label {
        background-color: transparent;
        padding: 12px 10px;
        border-radius: 6px;
        transition: all 0.2s;
        margin-bottom: 2px;
        color: #94a3b8 !important; /* Muted Grey default */
        font-weight: 500;
        font-size: 15px;
        cursor: pointer;
        display: flex;
        align-items: center;
    }
    
    /* Hover State */
    div[data-testid="stRadio"] label:hover {
        color: #ffffff !important;
        background-color: rgba(255, 255, 255, 0.05);
    }
    
    /* Active State */
    div[data-testid="stRadio"] label[data-checked="true"] {
        color: #ffffff !important; /* Bright White */
        font-weight: 600;
        background-color: transparent;
    }
    
    div[data-testid="stRadio"] div[role="radiogroup"] > label > div:first-child {
        display: none;
    }

    /* --- LOGO STYLING --- */
    .logo-container {
        display: flex;
        align-items: center;
        gap: 12px;
        padding-top: 20px;
        padding-bottom: 20px;
    }
    .logo-box {
        width: 40px;
        height: 40px;
        background-color: #3b82f6;
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-weight: 700;
        font-size: 20px;
    }
    .logo-text {
        font-size: 18px;
        font-weight: 600;
        color: #f1f5f9;
    }

    /* --- MAIN CONTENT STYLING --- */
    .main-card { 
        background-color: white; 
        border-radius: 12px; 
        padding: 25px; 
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05); 
        border: 1px solid #e2e8f0; 
        margin-bottom: 20px; 
    }
    
    .metric-card {
        background-color: white;
        border-radius: 12px;
        padding: 25px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        border: 1px solid #e2e8f0;
        height: 140px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    .metric-value { font-size: 2.5rem; font-weight: 700; color: #0f172a; margin: 0; }
    .metric-label { font-size: 0.8rem; font-weight: 700; color: #64748b; text-transform: uppercase; margin-top: 5px; }
    
    .pass-badge { color: #166534; background: #dcfce7; padding: 4px 8px; border-radius: 4px; font-weight: 600; font-size: 12px; }
    .fail-badge { color: #991b1b; background: #fee2e2; padding: 4px 8px; border-radius: 4px; font-weight: 600; font-size: 12px; }

    .stButton>button { 
        background-color: #2563eb; 
        color: white; 
        border-radius: 6px; 
        border: none; 
        font-weight: 600;
        padding: 0.5rem 1rem;
    }
    .stButton>button:hover { background-color: #1d4ed8; }
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

# --- 4. PAGE: DATA INGESTION & RUN MAPPING ---
def render_ingestion():
    st.markdown('<div style="color:#64748b; margin-bottom:5px;">Workspace / Data Ingestion</div>', unsafe_allow_html=True)
    st.title("Data Ingestion")
    
    if 'p_df' not in st.session_state: st.session_state['p_df'] = None
    if 'u_df' not in st.session_state: st.session_state['u_df'] = None

    # UPLOAD SECTION
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

    # ACTION BUTTON (MOVED HERE)
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
            
            st.write("Generating Audit Plans...")
            bar = st.progress(0)
            
            # Simple loop for demo (Top 5 rows)
            for i in range(len(p_df.head(5))):
                bar.progress((i+1)/5)
                scores = util.cos_sim(p_emb[i], u_emb)[0]
                best_idx = torch.topk(scores, k=1)[1][0].item()
                ctrl = u_df.iloc[best_idx]
                
                # Gemini Call
                plan = call_gemini(f"Role: Auditor. Control: {ctrl['Control_Text']}. Policy: {p_df.iloc[i]['Policy_Text']}. Task: Write concise test procedure.")
                
                # DB Save
                db_save_mapping(ctrl['Control_ID'], ctrl['Control_Text'], p_df.iloc[i]['Policy_Text'], plan)
            
            st.success("Mapping Complete & Saved to Database!")
            st.info("Navigate to **Control Mapping** to view the results.")

# --- 5. PAGE: CONTROL MAPPING (VIEW ONLY) ---
def render_mapping_view():
    st.markdown('<div style="color:#64748b; margin-bottom:5px;">Workspace / Control Mapping</div>', unsafe_allow_html=True)
    st.title("Control Mapping View")
    
    df = db_get_mappings()
    
    if df.empty:
        st.warning("⚠️ No mappings found. Please go to **Data Ingestion** and run the AI Mapping first.")
        return

    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    st.subheader(f"Mapped Relationships ({len(df)} Controls)")
    st.write("The AI has aligned your Internal Policies with the Regulatory Framework:")
    
    # Display simplified table of mappings
    display_df = df[['control_id', 'control_text', 'policy_text']].rename(columns={
        'control_id': 'Control ID',
        'control_text': 'UCF Control Requirement',
        'policy_text': 'Mapped Internal Policy'
    })
    
    st.dataframe(display_df, use_container_width=True, hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)

# --- 6. PAGE: EVALUATION GUIDANCE ---
def render_evaluation():
    st.markdown('<div style="color:#64748b; margin-bottom:5px;">Workspace / Evaluation Guidance</div>', unsafe_allow_html=True)
    st.title("Evaluation Guidance")
    
    df = db_get_mappings()
    if df.empty: st.info("No mappings found. Please run Data Ingestion first."); return

    st.markdown(f"**Total Controls:** {len(df)}")
    
    for i, row in df.iterrows():
        icon = "🟢" if row['status'] == 'PASS' else "🔴" if row['status'] == 'FAIL' else "⚪"
        with st.expander(f"{icon} {row['control_id']} (Status: {row['status']})"):
            st.markdown(f"**Control:** {row['control_text']}")
            st.markdown(f"**Policy:** {row['policy_text']}")
            st.divider()
            st.markdown("**🤖 AI Test Plan:**")
            st.info(row['ai_plan'])

# --- 7. PAGE: AUDIT AGENTS ---
def render_agents():
    st.markdown('<div style="color:#64748b; margin-bottom:5px;">Workspace / Audit Agents</div>', unsafe_allow_html=True)
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
        badge_class = "pass-badge" if record['status'] == 'PASS' else "fail-badge"
        st.markdown(f"""<div style="margin-top:20px; padding:15px; border:1px solid #e2e8f0; border-radius:8px;">
            <span class="{badge_class}">{record['status']}</span> <span style="margin-left:10px;">{record['last_result']}</span>
        </div>""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown(f"### 📜 Audit History")
    history = db_get_history(selected_id)
    if not history.empty: st.dataframe(history, use_container_width=True)

# --- 8. PAGE: DASHBOARD ---
def render_dashboard():
    # Breadcrumb & Title
    st.markdown('<div style="color:#64748b; margin-bottom:5px;">Workspace / Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<h1 style="color:#0f172a; font-weight:800; margin-bottom:20px;">Executive Dashboard</h1>', unsafe_allow_html=True)
    
    df = db_get_mappings()
    if df.empty: st.info("No data available. Please go to Data Ingestion."); return

    # Stats
    total = len(df)
    passed = len(df[df['status'] == 'PASS'])
    failed = len(df[df['status'] == 'FAIL'])
    untested = len(df[df['status'] == 'Untested'])
    score = int((passed / total) * 100) if total > 0 else 0

    # Metrics
    c1, c2, c3, c4 = st.columns(4)
    
    def metric_html(val, lbl, color="#0f172a"):
        return f"""<div class="metric-card">
            <h1 class="metric-value" style="color: {color}">{val}</h1>
            <p class="metric-label">{lbl}</p>
        </div>"""

    c1.markdown(metric_html(f"{score}%", "COMPLIANCE SCORE"), unsafe_allow_html=True)
    c2.markdown(metric_html(passed, "PASSING", "#22c55e"), unsafe_allow_html=True)
    c3.markdown(metric_html(failed, "FAILING", "#ef4444"), unsafe_allow_html=True)
    c4.markdown(metric_html(untested, "UNTESTED", "#64748b"), unsafe_allow_html=True)

    st.write("")
    st.write("")

    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Control Status")
        st.markdown('<div class="main-card">', unsafe_allow_html=True)
        chart_data = pd.DataFrame({'Status': ['Passing', 'Failing', 'Untested'], 'Count': [passed, failed, untested]})
        fig = px.pie(chart_data, values='Count', names='Status', hole=0.5, 
                     color='Status', 
                     color_discrete_map={'Passing':'#22c55e', 'Failing':'#ef4444', 'Untested':'#cbd5e1'})
        fig.update_layout(margin=dict(t=0, b=0, l=0, r=0), height=250)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown("### Risk by Domain")
        st.markdown('<div class="main-card">', unsafe_allow_html=True)
        domain_data = pd.DataFrame({
            "Domain": ["Access", "Encryption", "Logging", "Network"],
            "Risk": [10, 80, 20, 40]
        })
        fig2 = px.bar(domain_data, x="Domain", y="Risk", color="Risk", color_continuous_scale="Reds")
        fig2.update_layout(margin=dict(t=0, b=0, l=0, r=0), height=250, showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

# --- 9. SIDEBAR NAVIGATION ---
with st.sidebar:
    st.markdown("""
    <div class="logo-container">
        <div class="logo-box">U</div>
        <div class="logo-text">UCF Platform</div>
    </div>
    <div class="menu-header">Menu</div>
    """, unsafe_allow_html=True)
    
    # Custom Menu Order
    page = st.radio("Navigation", [
        "Dashboard", 
        "Data Ingestion", 
        "Control Mapping", 
        "Evaluation Guidance", 
        "Audit Agents"
    ], label_visibility="collapsed")
    
    st.write("---")
    st.caption("v3.3 (Production)")
    if st.button("Reset DB"):
        c.execute("DELETE FROM mappings"); c.execute("DELETE FROM audit_logs"); conn.commit()
        st.rerun()

# --- 10. ROUTING ---
if page == "Dashboard": render_dashboard()
elif page == "Data Ingestion": render_ingestion()
elif page == "Control Mapping": render_mapping_view()
elif page == "Evaluation Guidance": render_evaluation()
elif page == "Audit Agents": render_agents()
