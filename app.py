import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch
import google.generativeai as genai
import json
import sqlite3
import datetime
import random

# --- 1. CONFIG & SETUP ---
st.set_page_config(page_title="UCF Enterprise Platform", page_icon="🛡️", layout="wide")

# Initialize SQLite
conn = sqlite3.connect('grc_cloud.db', check_same_thread=False)
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS mappings (control_id TEXT, control_text TEXT, policy_text TEXT, ai_plan TEXT, status TEXT)''')
c.execute('''CREATE TABLE IF NOT EXISTS audit_logs (timestamp TEXT, control_id TEXT, evidence_source TEXT, result TEXT, reason TEXT)''')
conn.commit()

# --- 2. CSS STYLING ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; color: #0f172a; }
    .stApp { background-color: #f8fafc; }
    section[data-testid="stSidebar"] { background-color: #020617; }
    section[data-testid="stSidebar"] * { color: #94a3b8 !important; }
    .main-card { background-color: white; border-radius: 12px; padding: 25px; border: 1px solid #e2e8f0; box-shadow: 0 1px 3px rgba(0,0,0,0.05); margin-bottom: 20px; }
    .metric-val { font-size: 2.5rem; font-weight: 700; color: #0f172a; }
    .metric-lbl { color: #64748b; font-size: 0.9rem; font-weight: 600; text-transform: uppercase; }
    .agent-terminal { background-color: #1e1e1e; color: #00ff00; font-family: 'Courier New', monospace; padding: 15px; border-radius: 5px; font-size: 13px; }
</style>
""", unsafe_allow_html=True)

# --- 3. FUNCTIONS ---
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

def call_gemini(prompt):
    try:
        if "GOOGLE_API_KEY" in st.secrets:
            genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
            
            # Smart Model Selection
            candidate_models = ['gemini-1.5-flash-latest', 'gemini-1.5-flash', 'gemini-pro']
            for model_name in candidate_models:
                try:
                    model = genai.GenerativeModel(model_name)
                    response = model.generate_content(prompt)
                    return response.text
                except:
                    continue
            return "❌ Error: No working Gemini model found for this key."
        else:
            return "Error: GOOGLE_API_KEY missing in Secrets."
    except Exception as e:
        return f"AI Error: {e}"

# --- ENTERPRISE PROMPTS (THE UPGRADE) ---
def generate_ai_guidance(control_text, policy_text):
    prompt = f"""
    You are a Senior IT Security Auditor (CISA).
    
    INPUT:
    - Control: "{control_text}"
    - Policy: "{policy_text}"
    
    TASK:
    Create a rigorous Audit Test Plan.
    
    OUTPUT FORMAT (Markdown):
    **1. Objective:** [One sentence summary]
    **2. Technical Check:** [Specific API/Config to inspect]
    **3. Success Criteria:** [Exact condition for PASS]
    **4. Evidence:** [Required Artifact]
    """
    return call_gemini(prompt)

def run_ai_audit(control_text, evidence_json):
    prompt = f"""
    You are an Automated Compliance Engine.
    
    CONTEXT:
    - Requirement: "{control_text}"
    - Evidence: {json.dumps(evidence_json)}
    
    LOGIC:
    1. Identify key/value in JSON.
    2. Compare against Requirement.
    
    OUTPUT:
    - Start with "PASS" or "FAIL".
    - Follow with a hyphen (-).
    - Provide technical justification.
    """
    return call_gemini(prompt)

# --- DB HELPERS (THE FIX IS HERE) ---
def save_mapping(cid, ctxt, ptxt, plan):
    c.execute("INSERT INTO mappings VALUES (?, ?, ?, ?, ?)", (cid, ctxt, ptxt, plan, 'Untested'))
    conn.commit()

def save_audit(cid, src, res, reason):
    c.execute("UPDATE mappings SET status = ? WHERE control_id = ?", (res, cid))
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute("INSERT INTO audit_logs VALUES (?, ?, ?, ?, ?)", (ts, cid, src, res, reason))
    conn.commit()

# --- 4. PAGE ROUTING ---
def render_ingestion():
    st.title("📂 Data Ingestion")
    st.info("Upload CSV files to begin.")
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="main-card"><h3>1. Policies</h3>', unsafe_allow_html=True)
        f = st.file_uploader("Internal Policies CSV", key="p", type=["csv"])
        if f: 
            try:
                st.session_state['p_df'] = pd.read_csv(f, encoding='utf-8-sig')
                st.success(f"Loaded {len(st.session_state['p_df'])} rows")
            except: st.error("Invalid CSV")
        st.markdown('</div>', unsafe_allow_html=True)
        
    with c2:
        st.markdown('<div class="main-card"><h3>2. Controls</h3>', unsafe_allow_html=True)
        f = st.file_uploader("Regulatory Controls CSV", key="u", type=["csv"])
        if f: 
            try:
                st.session_state['u_df'] = pd.read_csv(f, encoding='utf-8-sig')
                st.success(f"Loaded {len(st.session_state['u_df'])} rows")
            except: st.error("Invalid CSV")
        st.markdown('</div>', unsafe_allow_html=True)

def render_mapping():
    st.title("🔗 Control Mapping")
    if 'p_df' not in st.session_state or 'u_df' not in st.session_state: 
        st.warning("Please upload CSVs in Ingestion tab."); return
    
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    if st.button("🚀 Run AI Mapping", use_container_width=True):
        embedder = load_embedding_model()
        p_df, u_df = st.session_state['p_df'], st.session_state['u_df']
        
        # Validation
        if 'Policy_Text' not in p_df.columns or 'Control_Text' not in u_df.columns:
            st.error("CSV columns mismatch. Need 'Policy_Text' and 'Control_Text'.")
            return

        p_emb = embedder.encode(p_df['Policy_Text'].head(5).astype(str).tolist(), convert_to_tensor=True)
        u_emb = embedder.encode(u_df['Control_Text'].head(5).astype(str).tolist(), convert_to_tensor=True)
        
        c.execute("DELETE FROM mappings"); conn.commit()
        prog = st.progress(0)
        
        for i in range(len(p_df.head(5))):
            prog.progress((i+1)/5)
            scores = util.cos_sim(p_emb[i], u_emb)[0]
            best_idx = torch.topk(scores, k=1)[1][0].item()
            ctrl = u_df.iloc[best_idx]
            
            # Using the NEW Enterprise Prompt
            plan = generate_ai_guidance(ctrl['Control_Text'], p_df.iloc[i]['Policy_Text'])
            
            save_mapping(ctrl['Control_ID'], ctrl['Control_Text'], p_df.iloc[i]['Policy_Text'], plan)
        st.success("Mapping Saved to DB")
    st.markdown('</div>', unsafe_allow_html=True)
    
    df = pd.read_sql("SELECT control_id, control_text, ai_plan FROM mappings", conn)
    if not df.empty: 
        for i, row in df.iterrows():
            with st.expander(f"{row['control_id']} - Audit Plan"):
                st.markdown(row['ai_plan'])

def render_testing():
    st.title("🤖 Automated Control Testing")
    df = pd.read_sql("SELECT * FROM mappings", conn)
    if df.empty: st.warning("Map controls first."); return
    
    c1, c2 = st.columns([1, 2])
    with c1:
        st.markdown('<div class="main-card">', unsafe_allow_html=True)
        sel = st.selectbox("Select Control", df['control_id'])
        row = df[df['control_id'] == sel].iloc[0]
        st.info(f"**Req:** {row['control_text']}")
        st.markdown('</div>', unsafe_allow_html=True)
        
    with c2:
        st.markdown('<div class="main-card"><h3>📡 Evidence Connector</h3>', unsafe_allow_html=True)
        f = st.file_uploader("Upload Wiz/AWS JSON", type=['json'])
        if f and st.button("Run AI Audit"):
            try:
                evidence = json.load(f)
                st.markdown(f"<div class='agent-terminal'>{json.dumps(evidence, indent=2)[:500]}...</div>", unsafe_allow_html=True)
                
                with st.spinner("Analyzing Evidence..."):
                    # Using the NEW Enterprise Prompt
                    res = run_ai_audit(row['control_text'], evidence)
                    
                    status = "PASS" if "PASS" in res.upper() else "FAIL"
                    save_audit(sel, "JSON Upload", status, res)
                    
                    if status == "PASS": st.success(f"✅ {res}")
                    else: st.error(f"❌ {res}")
            except Exception as e:
                st.error(f"Invalid JSON file: {e}")
        st.markdown('</div>', unsafe_allow_html=True)

def render_dashboard():
    st.title("📊 Live Compliance Dashboard")
    df = pd.read_sql("SELECT status FROM mappings", conn)
    if df.empty: st.info("No data yet."); return
    
    total = len(df); passed = len(df[df['status']=='PASS']); failed = len(df[df['status']=='FAIL'])
    score = int((passed/total)*100) if total > 0 else 0
    
    c1, c2, c3 = st.columns(3)
    c1.markdown(f'<div class="main-card"><div class="metric-val">{score}%</div><div class="metric-lbl">Score</div></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="main-card"><div class="metric-val" style="color:green">{passed}</div><div class="metric-lbl">Passing</div></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="main-card"><div class="metric-val" style="color:red">{failed}</div><div class="metric-lbl">Failing</div></div>', unsafe_allow_html=True)
    
    st.subheader("Audit Logs")
    logs = pd.read_sql("SELECT * FROM audit_logs ORDER BY timestamp DESC", conn)
    st.dataframe(logs, use_container_width=True)

# --- NAVIGATION ---
with st.sidebar:
    st.header("UCF Platform")
    page = st.radio("Menu", ["Dashboard", "Ingestion", "Mapping", "Control Testing"])
    if st.button("Reset DB"):
        c.execute("DELETE FROM mappings"); c.execute("DELETE FROM audit_logs"); conn.commit()
        st.rerun()

if page == "Dashboard": render_dashboard()
elif page == "Ingestion": render_ingestion()
elif page == "Mapping": render_mapping()
elif page == "Control Testing": render_testing()
