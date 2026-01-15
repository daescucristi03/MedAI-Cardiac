import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tempfile
import os
import sys
import time
from datetime import datetime

# Add project root to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

# Import Modules
from src.modules.ui_style import setup_page
from src.modules.database import (
    save_patient_record, get_patient_history, 
    register_doctor, authenticate_doctor, get_doctor_by_username,
    add_patient, find_patient, get_patient_details, get_all_patients,
    update_doctor_profile, get_doctor_activity_log, update_patient
)
from src.modules.report import generate_pdf
from src.modules.ecg_processor import (
    load_model, generate_advanced_ecg, calculate_metrics, 
    predict_risk, compute_saliency
)

# --- Icons URLs (Flat Style) ---
ICON_HEART = "https://cdn-icons-png.flaticon.com/512/2966/2966486.png" 
ICON_DOCTOR = "https://cdn-icons-png.flaticon.com/512/3304/3304567.png" 
ICON_PATIENT = "https://img.icons8.com/ios-glyphs/90/ffffff/user-male-circle.png" 
ICON_DASHBOARD = "https://cdn-icons-png.flaticon.com/512/9055/9055360.png" 
ICON_CHECK = "https://cdn-icons-png.flaticon.com/512/148/148767.png" 
ICON_WARN = "https://cdn-icons-png.flaticon.com/512/497/497738.png" 
ICON_SAVE = "https://cdn-icons-png.flaticon.com/512/2874/2874091.png" 
ICON_EDIT = "https://cdn-icons-png.flaticon.com/512/1159/1159633.png" 

# --- Setup ---
setup_page()

# --- Helper: Custom Alerts ---
def display_alert(type, message):
    icon = ICON_CHECK if type == "success" else ICON_WARN
    color = "#2ecc71" if type == "success" else "#e74c3c"
    bg_color = "#0e2b16" if type == "success" else "#2c0b0e"
    
    st.markdown(f"""
        <div style="background-color: {bg_color}; padding: 15px; border-radius: 10px; border-left: 5px solid {color}; display: flex; align-items: center; margin-bottom: 10px;">
            <img src="{icon}" width="24" style="margin-right: 15px;">
            <span style="color: {color}; font-weight: bold; font-size: 16px;">{message}</span>
        </div>
    """, unsafe_allow_html=True)

# --- Session State Init ---
def init_session():
    if 'authenticated' not in st.session_state:
        params = st.query_params
        if "doctor_user" in params:
            username = params["doctor_user"]
            doctor_data = get_doctor_by_username(username)
            if doctor_data:
                st.session_state['authenticated'] = True
                st.session_state['doctor_data'] = doctor_data
            else:
                st.session_state['authenticated'] = False
        else:
            st.session_state['authenticated'] = False

    if 'doctor_data' not in st.session_state:
        st.session_state['doctor_data'] = {"full_name": "", "specialty": ""}
    if 'active_patient_cnp' not in st.session_state:
        st.session_state['active_patient_cnp'] = None
    if 'view' not in st.session_state:
        st.session_state['view'] = 'dashboard'
    if 'current_signal' not in st.session_state:
        st.session_state['current_signal'] = None
    if 'current_time' not in st.session_state:
        st.session_state['current_time'] = None
    if 'analysis_done' not in st.session_state:
        st.session_state['analysis_done'] = False
    if 'last_uploaded_file' not in st.session_state:
        st.session_state['last_uploaded_file'] = None

init_session()

# --- Authentication Logic ---
def login_page():
    col_l, col_c, col_r = st.columns([2, 1, 2])
    with col_c:
        st.image(ICON_HEART, width=80)
    
    st.markdown("<h1 style='text-align: center;'>MedAI Portal Access</h1>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 1.5, 1])
    
    with col2:
        tab1, tab2 = st.tabs(["Login", "Register New Doctor"])
        with tab1:
            with st.form("login_form"):
                st.markdown("### Doctor Login")
                username = st.text_input("Username")
                password = st.text_input("Password", type="password")
                submit = st.form_submit_button("Login", type="primary", use_container_width=True)
                if submit:
                    success, doctor_data = authenticate_doctor(username, password)
                    if success:
                        st.session_state['authenticated'] = True
                        st.session_state['doctor_data'] = doctor_data
                        st.query_params["doctor_user"] = username
                        display_alert("success", f"Welcome back, Dr. {doctor_data['full_name']}!")
                        time.sleep(0.5)
                        st.rerun()
                    else:
                        display_alert("error", "Invalid username or password")
        with tab2:
            with st.form("register_form"):
                st.markdown("### New Account")
                new_user = st.text_input("Choose Username")
                new_pass = st.text_input("Choose Password", type="password")
                full_name = st.text_input("Full Name")
                specialty = st.selectbox("Specialty", ["Cardiology", "Internal Medicine", "Emergency"])
                reg_submit = st.form_submit_button("Register", type="primary", use_container_width=True)
                if reg_submit:
                    if new_user and new_pass and full_name:
                        success, msg = register_doctor(new_user, new_pass, full_name, specialty)
                        if success: display_alert("success", "Account created! Please login.")
                        else: display_alert("error", f"Registration failed: {msg}")

# --- Helper: Sidebar Doctor Info ---
def render_sidebar_doctor_info():
    doc = st.session_state['doctor_data']
    st.markdown(f"""
        <div style="background-color: #2C3E50; padding: 15px; border-radius: 10px; margin-bottom: 20px; border-left: 4px solid #2F80ED;">
            <div style="font-size: 12px; color: #E0F7FA; text-transform: uppercase; letter-spacing: 1px;">Logged in as</div>
            <div style="font-weight: bold; font-size: 16px; color: #56CCF2; margin-top: 5px;">Dr. {doc['full_name']}</div>
            <div style="font-size: 13px; color: #2F80ED;">{doc['specialty']}</div>
            <div style="font-size: 11px; color: #56CCF2; margin-top: 5px;">● Online</div>
        </div>
    """, unsafe_allow_html=True)
    
    if st.button("My Profile", type="secondary", use_container_width=True):
        st.session_state['view'] = 'doctor_profile'
        st.rerun()

# --- Doctor Profile View ---
def doctor_profile_view():
    doc = st.session_state['doctor_data']
    
    with st.sidebar:
        c1, c2 = st.columns([1, 3])
        with c1: st.image(ICON_HEART, width=50)
        with c2: st.markdown("<h2 style='text-align: left; margin: 0; padding-top: 10px;'>MedAI Control</h2>", unsafe_allow_html=True)
        
        render_sidebar_doctor_info()
        
        if st.button("Back to Dashboard", type="secondary", use_container_width=True):
            st.session_state['view'] = 'dashboard'
            st.rerun()
            
        if st.button("Logout", type="secondary", use_container_width=True):
            st.session_state['authenticated'] = False
            st.query_params.clear()
            st.rerun()

    st.title(f"👨‍⚕️ Profile: Dr. {doc['full_name']}")
    
    tab_edit, tab_activity = st.tabs(["Edit Profile", "Activity Log"])
    
    with tab_edit:
        with st.form("edit_profile_form"):
            new_name = st.text_input("Full Name", value=doc['full_name'])
            new_specialty = st.selectbox("Specialty", ["Cardiology", "Internal Medicine", "Emergency"], index=["Cardiology", "Internal Medicine", "Emergency"].index(doc['specialty']))
            new_pass = st.text_input("New Password (leave blank to keep current)", type="password")
            
            if st.form_submit_button("Update Profile", type="primary"):
                success, msg = update_doctor_profile(doc['username'], new_name, new_specialty, new_pass if new_pass else None)
                if success:
                    display_alert("success", msg)
                    st.session_state['doctor_data']['full_name'] = new_name
                    st.session_state['doctor_data']['specialty'] = new_specialty
                    time.sleep(1)
                    st.rerun()
                else:
                    display_alert("error", msg)
    
    with tab_activity:
        logs = get_doctor_activity_log(doc['full_name'])
        if logs:
            df_logs = pd.DataFrame(logs)
            total_scans = len(df_logs)
            high_risk = len(df_logs[df_logs['risk_score'] > 0.5])
            
            c1, c2 = st.columns(2)
            c1.metric("Total Consultations", total_scans)
            c2.metric("High Risk Cases Detected", high_risk)
            
            st.dataframe(df_logs[['timestamp', 'patient_cnp', 'risk_score', 'diagnosis']], use_container_width=True)
        else:
            st.info("No activity recorded yet.")

# --- Patient Edit View ---
def patient_edit_view(cnp):
    pat = get_patient_details(cnp)
    if not pat:
        st.error("Patient not found.")
        st.session_state['view'] = 'dashboard'
        st.rerun()

    with st.sidebar:
        c1, c2 = st.columns([1, 3])
        with c1: st.image(ICON_HEART, width=50)
        with c2: st.markdown("<h2 style='text-align: left; margin: 0; padding-top: 10px;'>MedAI Control</h2>", unsafe_allow_html=True)
        render_sidebar_doctor_info()
        
        if st.button("Cancel Edit", type="secondary", use_container_width=True):
            st.session_state['view'] = 'dashboard'
            st.rerun()

    st.title(f"✏️ Edit Patient: {pat['first_name']} {pat['last_name']}")
    
    with st.form("edit_patient_form"):
        c1, c2 = st.columns(2)
        with c1:
            fname = st.text_input("First Name", value=pat['first_name'])
            age = st.number_input("Age", 0, 120, pat['age'])
        with c2:
            lname = st.text_input("Last Name", value=pat['last_name'])
            sex = st.selectbox("Sex", ["M", "F"], index=["M", "F"].index(pat['sex']))
            
        history = st.text_area("Medical History Notes", value=pat['medical_history_notes'])
        
        if st.form_submit_button("Update Patient Record", type="primary", use_container_width=True):
            success, msg = update_patient(cnp, fname, lname, age, sex, history)
            if success:
                display_alert("success", msg)
                time.sleep(1)
                st.session_state['view'] = 'patient_profile'
                st.rerun()
            else:
                display_alert("error", msg)

# --- Patient Profile View ---
def patient_profile_view(cnp):
    pat = get_patient_details(cnp)
    if not pat:
        st.error("Patient not found.")
        if st.button("Back to Dashboard"):
            st.session_state['view'] = 'dashboard'
            st.rerun()
        return

    # --- SIDEBAR: Patient Details ---
    with st.sidebar:
        c1, c2 = st.columns([1, 3])
        with c1: st.image(ICON_HEART, width=50)
        with c2: st.markdown("<h2 style='text-align: left; margin: 0; padding-top: 10px;'>MedAI Control</h2>", unsafe_allow_html=True)
            
        render_sidebar_doctor_info()
        
        st.markdown("---")
        st.markdown("### Active Patient")
        st.markdown(f"""
            <div style="display: flex; align-items: center; background-color: #2C3E50; padding: 10px; border-radius: 10px; margin-bottom: 15px; border: 1px solid #2F80ED;">
                <img src="{ICON_PATIENT}" width="40" style="margin-right: 15px;">
                <div>
                    <div style="font-weight: bold; font-size: 16px; color: #E0F7FA;">{pat['first_name']} {pat['last_name']}</div>
                    <div style="font-size: 12px; color: #56CCF2;">CNP: {pat['cnp']}</div>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        st.write(f"**Age:** {pat['age']}")
        st.write(f"**Sex:** {pat['sex']}")
        st.write(f"**Registered:** {str(pat['created_at'])[:10]}")
        
        with st.expander("Medical History Notes", expanded=True):
            st.caption(pat['medical_history_notes'] or "No notes available.")
            
        st.markdown("---")
        if st.button("Back to Dashboard", type="secondary", use_container_width=True):
            st.session_state['view'] = 'dashboard'
            st.session_state['active_patient_cnp'] = None
            st.session_state['current_signal'] = None 
            st.session_state['analysis_done'] = False
            st.rerun()
            
        if st.button("Logout", type="secondary", use_container_width=True):
            st.session_state['authenticated'] = False
            st.query_params.clear()
            st.rerun()

    # --- MAIN CONTENT ---
    col_title, col_edit = st.columns([4, 1])
    with col_title:
        st.markdown(f"## Analysis Dashboard: {pat['first_name']} {pat['last_name']}")
    with col_edit:
        if st.button("✏️ Edit Details", type="secondary"):
            st.session_state['view'] = 'patient_edit'
            st.rerun()

    tab_new, tab_hist = st.tabs(["New Analysis", "Medical History"])

    # --- TAB 1: New Analysis ---
    with tab_new:
        col_cfg, col_main = st.columns([1, 2])
        
        with col_cfg:
            st.markdown("#### Signal Source")
            mode = st.radio("Source", ["Simulator", "Upload File"], key="source_mode")
            
            if mode == "Simulator":
                heart_rate = st.slider("Heart Rate", 40, 140, 70)
                noise_level = st.slider("Noise", 0.0, 0.2, 0.02)
                # st_disp = st.slider("ST Displacement", -0.5, 1.0, 0.0, step=0.1)
                st_disp = 0.0
                t_amp = st.slider("T-Wave Amp", -0.5, 0.8, 0.25, step=0.05)
                # sensitivity = st.slider("AI Sensitivity", 1.0, 5.0, 2.5, step=0.1)
                sensitivity = 1.0

                if st.button("Generate Signal", type="primary", use_container_width=True):
                    signal, time_axis = generate_advanced_ecg(heart_rate, noise_level, st_disp, t_amp)
                    st.session_state['current_signal'] = signal
                    st.session_state['current_time'] = time_axis
                    st.session_state['analysis_done'] = False # Reset analysis
                    st.rerun()
            else:
                uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
                sensitivity = 1.0
                if uploaded_file:
                    # Check if file changed
                    if st.session_state.get('last_uploaded_file') != uploaded_file.name:
                        try:
                            uploaded_file.seek(0) # Reset pointer
                            df = pd.read_csv(uploaded_file)
                            signal = df.iloc[:, :12].values
                            if len(signal) > 5000: signal = signal[:5000]
                            st.session_state['current_signal'] = signal
                            st.session_state['current_time'] = np.linspace(0, len(signal)/500, len(signal))
                            st.session_state['analysis_done'] = False
                            st.session_state['last_uploaded_file'] = uploaded_file.name
                        except Exception as e: 
                            st.error(f"Invalid CSV: {e}")

        with col_main:
            if st.session_state['current_signal'] is not None:
                signal = st.session_state['current_signal']
                t = st.session_state['current_time']
                
                # Plot
                fig, ax = plt.subplots(figsize=(8, 2.5))
                ax.plot(t[:1000], signal[:1000, 6], color='#56CCF2', linewidth=1)
                ax.set_title("Lead V1 Preview", fontsize=10, color='#E0F7FA')
                ax.set_facecolor('#1F2D3D')
                fig.patch.set_facecolor('#1F2D3D')
                ax.tick_params(colors='#E0F7FA', labelsize=8)
                ax.spines['bottom'].set_color('#2F80ED')
                ax.spines['left'].set_color('#2F80ED')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                plt.tight_layout()
                st.pyplot(fig)
                
                # Diagnosis
                if st.button("Run AI Diagnosis", type="primary", use_container_width=True):
                    model = load_model()
                    if model:
                        with st.spinner("Analyzing..."):
                            risk = predict_risk(model, signal, sensitivity)
                            st.session_state['risk_score'] = risk
                            st.session_state['analysis_done'] = True
                            
                            # Save to DB
                            bpm, _ = calculate_metrics(signal)
                            diag = "High Risk" if risk > 0.5 else "Low Risk"
                            save_patient_record(cnp, bpm, risk, diag, st.session_state['doctor_data']['full_name'])
                            st.toast("Diagnosis saved to history!", icon="💾")
                            st.rerun()

                if st.session_state.get('analysis_done'):
                    score = st.session_state.get('risk_score', 0)
                    bpm, hrv = calculate_metrics(signal)
                    
                    # Calculate Confidence
                    confidence = abs(score - 0.5) * 2
                    
                    # Detailed Results
                    st.markdown("### 🔍 Diagnostic Results")
                    
                    if score > 0.5:
                        display_alert("error", f"HIGH RISK DETECTED: {score:.1%}")
                        st.caption("The model detected patterns consistent with Myocardial Infarction (ST-Elevation or T-Wave abnormalities). Immediate clinical correlation recommended.")
                    else:
                        display_alert("success", f"LOW RISK: {score:.1%}")
                        st.caption("Sinus rhythm appears normal. No significant ST-segment deviation detected.")
                    
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Risk Probability", f"{score:.1%}", delta_color="inverse")
                    m2.metric("Heart Rate", f"{int(bpm)} BPM")
                    m3.metric("HR Variability", f"{int(hrv)} ms")
                    
                    st.write(f"Model Confidence: **{int(confidence * 100)}%**")
                    st.progress(int(confidence * 100))
                    
                    st.divider()
                    
                    # PDF Download
                    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
                        fig.savefig(tmpfile.name)
                        plot_path = tmpfile.name
                        
                    pdf_bytes = generate_pdf(pat, score, bpm, hrv, plot_path, st.session_state['doctor_data']['full_name'])
                    st.download_button(
                        label="Download Report",
                        data=pdf_bytes,
                        file_name=f"report_{cnp}.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
            else:
                st.info("Generate or Upload a signal to start.")

    # --- TAB 2: History ---
    with tab_hist:
        history = get_patient_history(cnp)
        if history:
            hist_df = pd.DataFrame(history)
            st.dataframe(hist_df[['timestamp', 'doctor', 'heart_rate', 'risk_score', 'diagnosis']], use_container_width=True)
            st.markdown("#### Risk Evolution")
            st.line_chart(hist_df.set_index('timestamp')['risk_score'])
        else:
            st.info("No history found for this patient.")

# --- Main Dashboard (Patient List) ---
def main_dashboard():
    with st.sidebar:
        c1, c2 = st.columns([1, 3])
        with c1: st.image(ICON_HEART, width=50)
        with c2: st.markdown("<h2 style='text-align: left; margin: 0; padding-top: 10px;'>MedAI Control</h2>", unsafe_allow_html=True)
            
        render_sidebar_doctor_info()
        
        if st.button("Logout", type="secondary", use_container_width=True):
            st.session_state['authenticated'] = False
            st.query_params.clear() # Clear URL params on logout
            st.rerun()
    
    st.title("Patient Management Dashboard")
    
    col_search, col_add = st.columns([2, 1])
    
    with col_search:
        st.subheader("Find Patient")
        search_term = st.text_input("Search by Name or CNP", placeholder="Type here...")
        
        if search_term:
            results = find_patient(search_term)
        else:
            results = get_all_patients(limit=20) 
            
        if results:
            for p in results:
                with st.container():
                    c1, c2 = st.columns([3, 1])
                    c1.markdown(f"**{p['first_name']} {p['last_name']}** (CNP: {p['cnp']})")
                    
                    # Buttons
                    b1, b2 = st.columns([1, 1])
                    if b1.button("Open Profile", key=f"open_{p['cnp']}", type="primary", use_container_width=True):
                        st.session_state['active_patient_cnp'] = p['cnp']
                        st.session_state['view'] = 'patient_profile'
                        st.session_state['current_signal'] = None 
                        st.session_state['analysis_done'] = False
                        st.rerun()
                    
                    if b2.button("Edit", key=f"edit_{p['cnp']}", type="secondary", use_container_width=True):
                        st.session_state['active_patient_cnp'] = p['cnp']
                        st.session_state['view'] = 'patient_edit'
                        st.rerun()
                        
                    st.divider()
        else:
            st.warning("No patients found.")
    
    with col_add:
        st.subheader("Register New")
        with st.form("add_patient_form"):
            cnp = st.text_input("CNP")
            fname = st.text_input("First Name")
            lname = st.text_input("Last Name")
            age = st.number_input("Age", 0, 120, 50)
            sex = st.selectbox("Sex", ["M", "F"])
            history = st.text_area("Notes")
            
            if st.form_submit_button("Add Patient", type="primary", use_container_width=True):
                if cnp and fname:
                    success, msg = add_patient(cnp, fname, lname, age, sex, history)
                    if success:
                        display_alert("success", msg)
                        st.session_state['active_patient_cnp'] = cnp
                        st.session_state['view'] = 'patient_profile'
                        st.session_state['current_signal'] = None
                        st.session_state['analysis_done'] = False
                        time.sleep(1)
                        st.rerun()
                    else:
                        display_alert("error", msg)
                else:
                    st.warning("Required fields missing.")

# --- App Entry Point ---
if st.session_state['authenticated']:
    if st.session_state['view'] == 'dashboard':
        main_dashboard()
    elif st.session_state['view'] == 'patient_profile':
        patient_profile_view(st.session_state['active_patient_cnp'])
    elif st.session_state['view'] == 'doctor_profile':
        doctor_profile_view()
    elif st.session_state['view'] == 'patient_edit':
        patient_edit_view(st.session_state['active_patient_cnp'])
else:
    login_page()
