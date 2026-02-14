import streamlit as st
import pandas as pd
import numpy as np
import joblib
import zipfile
import os
import plotly.graph_objects as go
from autogluon.tabular import TabularPredictor

# ==========================================
# 1. ตั้งค่าหน้าเว็บและธีม (NOMOS Style)
# ==========================================
st.set_page_config(
    page_title="SME Fin Health Check",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS: หัวข้อ = Kanit, เนื้อหา = Sarabun
st.markdown("""
<style>
    /* 1. นำเข้า Font จาก Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Kanit:wght@300;400;600;700&family=Sarabun:wght@300;400;500;700&display=swap');
    
    /* 2. กำหนด Font พื้นฐาน (เนื้อหา) เป็น Sarabun */
    html, body, [class*="css"], p, div, label, .stMarkdown, .stTextInput, .stNumberInput, .stSelectbox {
        font-family: 'Sarabun', sans-serif;
        color: #333333;
    }

    /* 3. กำหนด Font หัวข้อ (Header) เป็น Kanit */
    h1, h2, h3, h4, h5, h6, .stTitle {
        font-family: 'Kanit', sans-serif !important;
        font-weight: 600; /* ปรับความหนาให้ดูเด่น */
    }
    
    /* 4. ปรับแต่งปุ่มกด (Button) ให้เป็น Kanit เพื่อความสวยงาม */
    .stButton>button {
        font-family: 'Kanit', sans-serif !important;
        border-radius: 20px;
        border: 1px solid #333;
        color: #333;
        background-color: white;
        padding: 10px 24px;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #333;
        color: white;
        border-color: #333;
    }

    /* 5. ปรับแต่ง Class พิเศษ (จากโค้ดหน้า Landing Page) */
    .hero-text {
        font-family: 'Kanit', sans-serif;
        font-size: 3em;
        font-weight: 400; /* แก้ไข: ปรับจาก 700 เป็น 400 ให้ตัวบางลง */
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 20px;
    }
    .sub-hero {
        font-family: 'Sarabun', sans-serif; /* คำโปรยรอง */
        font-size: 1.5em;
        font-weight: 300;
        color: #555;
        text-align: center;
        margin-bottom: 40px;
    }
    .step-indicator {
        font-family: 'Kanit', sans-serif;
        text-align: center;
        color: #888;
        font-size: 0.9em;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. ระบบจัดการ Session State (เพื่อเปลี่ยนหน้า)
# ==========================================
if 'page' not in st.session_state:
    st.session_state.page = 'landing'
if 'inputs' not in st.session_state:
    st.session_state.inputs = {}
if 'results' not in st.session_state:
    st.session_state.results = {}

def navigate_to(page):
    st.session_state.page = page
    st.rerun()

# ==========================================
# 3. ฟังก์ชันโหลดโมเดล (Resource Loader)
# ==========================================
@st.cache_resource
def load_resources():
    # 3.1 โหลด Clustering Model
    if not os.path.exists('kmeans_behavior_model.joblib') or not os.path.exists('scaler_behavior.joblib'):
        st.error("ไม่พบไฟล์โมเดล Clustering (.joblib) กรุณาตรวจสอบ GitHub")
        st.stop()
        
    kmeans = joblib.load('kmeans_behavior_model.joblib')
    scaler = joblib.load('scaler_behavior.joblib')

    # 3.2 โหลด AutoGluon (รวมไฟล์ Zip)
    extract_path = './autogluon_model_extracted'
    combined_zip_name = 'full_model_combined.zip'

    if not os.path.exists(extract_path):
        part_files = sorted([f for f in os.listdir('.') if f.startswith('model_part_')])
        
        if not part_files:
            # กรณีไม่มีไฟล์ part (อาจจะรันในเครื่องที่มีโฟลเดอร์อยู่แล้ว หรือไม่มีไฟล์เลย)
            if not os.path.exists("Ag-20250201_135012"): # เช็คชื่อโฟลเดอร์โมเดลจริง
                 st.warning("ไม่พบไฟล์โมเดล AutoGluon (model_part_*.zip) กำลังทำงานในโหมด Demo...")
                 return kmeans, scaler, None, None 

        if part_files:
            with open(combined_zip_name, 'wb') as combined_file:
                for part in part_files:
                    with open(part, 'rb') as p:
                        combined_file.write(p.read())
            
            with zipfile.ZipFile(combined_zip_name, 'r') as zip_ref:
                zip_ref.extractall(extract_path)

    # ค้นหา path ของ predictor.pkl
    model_path = extract_path
    for root, dirs, files in os.walk(extract_path):
        if 'predictor.pkl' in files:
            model_path = root
            break
    
    try:
        predictor = TabularPredictor.load(model_path, require_py_version_match=False)
    except:
        predictor = None # กรณีโหลดไม่ได้จริงๆ

    # 3.3 โหลดข้อมูลดิบ (สำหรับ Imputation)
    try:
        df_raw = pd.read_excel('RawData2.xlsx')
    except:
        df_raw = pd.DataFrame() # กรณีไม่มีไฟล์

    return kmeans, scaler, predictor, df_raw

# โหลดทรัพยากร
kmeans_model, scaler_model, predictor_model, df_raw = load_resources()

# ==========================================
# 4. ส่วนแสดงผล (Page Views)
# ==========================================

# --- หน้าที่ 1: Landing Page ---
def show_landing():
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image("FinCheck.jpg", use_container_width=True) # ภาพประกอบแนวการเงิน
        st.markdown('<div class="hero-text">ตรวจเช็คแหล่งเงินทุน<br>ของคุณในเสี้ยวนาที</div>', unsafe_allow_html=True)
        st.markdown('<div class="sub-hero">รู้ทันสุขภาพการเงิน | ประเมิน DNA ธุรกิจ | ลดความเสี่ยง | รับคำแนะนำ</div>', unsafe_allow_html=True)
        
        c_btn1, c_btn2, c_btn3 = st.columns([1, 2, 1])
        with c_btn2:
            if st.button("🚀 เริ่มประเมินทันที", use_container_width=True):
                navigate_to('input_step1')
        
        st.markdown("---")
        st.caption("พัฒนาโดย: นายสมเกียรติ จูสวัสดิ์ | นิสิตปริญญาเอก หลักสูตรธุรกิจเทคโนโลยีและการจัดการนวัตกรรม จุฬาลงกรณ์มหาวิทยาลัย")

# --- หน้าที่ 2: Input Step 1 (DNA) ---
def show_input_step1():
    st.markdown('<div class="step-indicator">ขั้นตอนที่ 1/2: ค้นหา DNA ธุรกิจของท่าน</div>', unsafe_allow_html=True)
    st.markdown("### 🧬 ข้อมูลพฤติกรรมและการจัดการ")
    st.info("💡 โปรดทราบ: 0 = ไม่มี, 1 = น้อยที่สุด, 5 = มากที่สุด")

    # กำหนดตัวเลือกคะแนน 0-5
    score_options = [1-5]

    with st.form("form_step1"):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**ด้านการตลาดและแบรนด์**")
            # แก้ไข [1-5] เป็น score_options และปรับ index ให้เหมาะสม
            beh_mon = st.selectbox("ท่านให้ความสำคัญกับวินัยและการติดตามตรวจสอบการเงิน (BEH_MON)", score_options, index=3)
            brn_image = st.selectbox("ท่านให้ความสำคัญกับภาพลักษณ์องค์กร (BRN_IMAGE)", score_options, index=3)
            brn_brand = st.selectbox("การรับรู้และความน่าเชื่อถือของแบรนด์ของท่าน (BRN_BRAND)", score_options, index=3)
        
        with col2:
            st.markdown("**ด้านเทคโนโลยีและนโยบาย**")
            sav_virus = st.selectbox("การอัพเดทโปรแกรมป้องกันไวรัสเพื่อความปลอดภัย (SAV_VIRUS)", score_options, index=3)
            sav_pdpa = st.selectbox("ท่านปฏิบัติตามกฎหมาย PDPA (SAV_PDPA)", score_options, index=1)
            cri_pln = st.selectbox("ท่านมีแผนรองรับวิกฤตการณ์/ซ้อมหนีไฟ (CRI_PLN)", score_options, index=2)
            pol_ben = st.selectbox("ท่านได้รับประโยชน์จากนโยบายภาครัฐ (POL_BEN)", score_options, index=2)
            pol_adj = st.selectbox("ท่านสามารถปรับรูปแบบธุรกิจให้สอดคล้องนโยบายรัฐ (POL_ADJ)", score_options, index=3)

        # ปุ่ม Submit ต้องอยู่ย่อหน้าเดียวกับ col1, col2 (ภายใน with st.form)
        submitted = st.form_submit_button("ถัดไป >", type="primary", use_container_width=True)
        
        if submitted:
            # บันทึกค่าลง Session
            st.session_state.inputs.update({
                'BEH_MON': beh_mon, 'BRN_IMAGE': brn_image, 'BRN_BRAND': brn_brand,
                'SAV_VIRUS': sav_virus, 'SAV_PDPA': sav_pdpa, 'CRI_PLN': cri_pln,
                'POL_BEN': pol_ben, 'POL_ADJ': pol_adj
            })
            navigate_to('input_step2')

# --- หน้าที่ 3: Input Step 2 (Business Mgmt) ---
def show_input_step2():
    st.markdown('<div class="step-indicator">ขั้นตอนที่ 2/2: การจัดการธุรกิจและการเงิน</div>', unsafe_allow_html=True)
    st.markdown("### 💼 ข้อมูลการจัดการธุรกิจ")
    st.info("💡 ข้อมูลเหล่านี้ใช้พยากรณ์โอกาสเข้าถึงแหล่งเงินทุน")

    with st.form("form_step2"):
        col1, col2 = st.columns(2)
        with col1:
            prc_cfw = st.slider("ระดับกระแสเงินสดเพื่อประกอบธุรกิจและชำระหนี้ (PRC_CFW)", 0, 5, 3)
            cap_netw = st.slider("การใช้เครือข่าย/พันธมิตรธุรกิจ (CAP_NETW)", 0, 5, 3)
            eco_adt = st.slider("ความสามารถในการปรับตัวรับสถานการณ์เศรษฐกิจ (ECO_ADT)", 0, 5, 3)
            ecm_net = st.slider("การเข้าถึงเครือข่ายอินเตอร์เน็ตของกิจการ (ECM_NET)", 0, 5, 3)
        
        with col2:
            res_ch = st.slider("ความสามารถในการโต้ตอบลูกค้าผ่านช่องทางต่างๆ (RES_CH)", 0, 5, 3)
            # เพิ่มเติมตาม Feature Importance
            tmc_live = st.slider("ทักษะการ Live ขายสินค้า (TMC_LIVE)", 0, 5, 2)
            csr3 = st.radio("กิจการมีระบบกำจัดของเสีย (CSR3)", ["ไม่มี (0)", "มี (1)"])
            ohr_career = st.radio("กิจการมีเส้นทางอาชีพให้พนักงาน (OHR_CAREER)", ["ไม่มี (0)", "มี (1)"])
            
            # แปลงค่า Radio
            csr3_val = 1 if "มี" in csr3 else 0
            ohr_career_val = 1 if "มี" in ohr_career else 0

        col_b1, col_b2 = st.columns([1])
        with col_b1:
            if st.form_submit_button("< ย้อนกลับ"):
                navigate_to('input_step1')
        with col_b2:
            submitted = st.form_submit_button("🚀 ประเมินผลลัพธ์")
            
        if submitted:
            st.session_state.inputs.update({
                'PRC_CFW': prc_cfw, 'CAP_NETW': cap_netw, 'ECO_ADT': eco_adt,
                'ECM_NET': ecm_net, 'RES_CH': res_ch, 'TMC_LIVE': tmc_live,
                'CSR3': csr3_val, 'OHR_CAREER': ohr_career_val
            })
            process_results() # คำนวณผล
            navigate_to('dashboard')

# --- ฟังก์ชันประมวลผล (Processing Logic) ---
def process_results():
    inputs = st.session_state.inputs
    
    # 1. Clustering Logic
    cluster_features = ['BEH_MON', 'BRN_IMAGE', 'BRN_BRAND', 'SAV_VIRUS', 'SAV_PDPA', 'CRI_PLN', 'POL_BEN', 'POL_ADJ']
    cluster_vals = [inputs[f] for f in cluster_features]
    
    try:
        # Scale & Predict
        X_cluster = pd.DataFrame([cluster_vals], columns=cluster_features)
        X_scaled = scaler_model.transform(X_cluster)
        cluster_id = kmeans_model.predict(X_scaled)
    except:
        cluster_id = 0 # Default if model fails
        
    st.session_state.results['cluster_id'] = cluster_id

    # 2. Prediction Logic (AutoGluon)
    if predictor_model is not None and not df_raw.empty:
        # สร้าง Row ข้อมูลใหม่จากค่าเฉลี่ย (Imputation Strategy)
        pred_df = df_raw.iloc[0:1].copy().reset_index(drop=True)
        
        # แทนค่าเฉลี่ย/ฐานนิยมในคอลัมน์ที่ไม่ได้ถาม
        for c in df_raw.columns:
            if c not in inputs.keys() and c not in ['ID', 'target']:
                if df_raw[c].dtype == 'object':
                    pred_df[c] = df_raw[c].mode()
                else:
                    pred_df[c] = df_raw[c].mean()
        
        # ใส่ค่าที่รับมาจาก User
        for key, val in inputs.items():
            if key in pred_df.columns:
                pred_df[key] = val
        
        # เพิ่ม SIZ และ YER (สมมติค่า Default หรือถามเพิ่มได้ ถ้าจำเป็น)
        pred_df['SIZ'] = 1 # Default Small
        pred_df['YER'] = 10 # Default Established
        
        # Predict Class 1 Probability
        try:
            prob = predictor_model.predict_proba(pred_df).iloc[1]
        except:
            prob = 0.5 # Fallback
    else:
        # Logic จำลองกรณีไม่มีไฟล์โมเดล (สำหรับการแสดงผล Demo)
        score = inputs['PRC_CFW'] * 0.4 + inputs['CAP_NETW'] * 0.3 + inputs['BEH_MON'] * 0.3
        prob = 1 - (score / 5.0) # คะแนนเยอะ ความเสี่ยงน้อย
        
    st.session_state.results['risk_prob'] = prob

# --- หน้าที่ 4: Dashboard Results ---
def show_dashboard():
    cluster_id = st.session_state.results.get('cluster_id', 0)
    risk_prob = st.session_state.results.get('risk_prob', 0.5)
    
    # Mapping Names (Cluster 0, 1, 2 ตามผลวิเคราะห์ใน Source 7Feb26 และ Requirements)
    # Cluster 0: Active Marketer (เน้นการตลาด แต่เสี่ยงหลังบ้าน)
    # Cluster 1: Potential Starter (เปราะบาง/คะแนนต่ำ)
    # Cluster 2: Master Leader (เก่งรอบด้าน)
    
    cluster_info = {
        0: {"name": "Active Marketer (นักการตลาดไฟแรง)", "desc": "โดดเด่นด้านการสร้างแบรนด์และการตลาด แต่ต้องระวังระบบหลังบ้าน", "color": "#3498db"},
        1: {"name": "Potential Starter (นักสู้ผู้มีศักยภาพ)", "desc": "อยู่ในช่วงเริ่มต้นสร้างรากฐาน ต้องการการเสริมสร้างวินัยการเงิน", "color": "#f1c40f"},
        2: {"name": "Master Leader (ผู้นำระดับมาสเตอร์)", "desc": "มีความพร้อมรอบด้าน ทั้งการเงิน การจัดการ และเทคโนโลยี", "color": "#2ecc71"}
    }
    
    c_data = cluster_info.get(cluster_id, cluster_info)

    st.markdown("## 📊 ผลการประเมิน (Financial Access Dashboard)")
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"### 🧬 DNA ของท่าน: <span style='color:{c_data['color']}'>{c_data['name']}</span>", unsafe_allow_html=True)
        st.info(c_data['desc'])
        st.markdown("""
        **ลักษณะเด่น:**
        *   วิเคราะห์จากพฤติกรรม 8 ด้าน
        *   สะท้อนตัวตนและวัฒนธรรมองค์กร
        """)

    with col2:
        risk_score = risk_prob * 100
        
        # กำหนดสีตามความเสี่ยง (เขียว=เสี่ยงต่ำ/เข้าถึงง่าย, แดง=เสี่ยงสูง/เข้าถึงยาก)
        # Model Output: 1 = มีข้อจำกัด (เสี่ยงสูง), 0 = ไม่มีข้อจำกัด (เสี่ยงต่ำ)
        # ดังนั้น Score สูง = สีแดง
        if risk_score < 40:
            risk_color = "green"
            risk_text = "ความเสี่ยงต่ำ (Low Risk)"
        elif risk_score < 70:
            risk_color = "orange"
            risk_text = "ความเสี่ยงปานกลาง (Moderate Risk)"
        else:
            risk_color = "red"
            risk_text = "ความเสี่ยงสูง (High Risk)"

        st.markdown(f"### 🔮 ระดับความเสี่ยงการเข้าถึงแหล่งเงิน: **{risk_score:.1f}%**")
        
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = risk_score,
            domain = {'x': [1], 'y': [1]},
            title = {'text': f"<span style='font-size:0.8em;color:gray'>{risk_text}</span>"},
            gauge = {
                'axis': {'range': [None, 100], 'tickwidth': 1},
                'bar': {'color': "darkblue"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [6], 'color': "#2ecc71"},  # เขียว
                    {'range': [6, 7], 'color': "#f1c40f"}, # เหลือง
                    {'range': [7, 8], 'color': "#e74c3c"} # แดง
                ],
            }
        ))
        fig.update_layout(height=250, margin=dict(l=20,r=20,t=30,b=20))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    if st.button("ดูคำแนะนำเจาะลึก (Recommendation) >", type="primary", use_container_width=True):
        navigate_to('recommendation')

# --- หน้าที่ 5: Recommendations ---
def show_recommendation():
    st.markdown("## 🎯 คำแนะนำสำหรับท่าน (Recommendations)")
    cluster_id = st.session_state.results.get('cluster_id', 0)
    
    # Recommendation Logic (Personalized)
    recs = {
        0: { # Active Marketer
            "strength": "ท่านมีความเข้มแข็งในการสร้างแบรนด์และภาพลักษณ์องค์กร",
            "urgent": "สร้างมาตรการความปลอดภัยข้อมูล (PDPA) และแผนรองรับวิกฤตด่วน! ธนาคารมองว่านี่คือความเสี่ยงแฝง",
            "maintain": "รักษาฐานลูกค้าและการตลาดออนไลน์ให้ต่อเนื่อง"
        },
        1: { # Potential Starter
            "strength": "ท่านมีความยืดหยุ่นและโอกาสในการเริ่มต้นวางระบบใหม่ที่ถูกต้อง",
            "urgent": "เริ่มทำบัญชีรายรับ-รายจ่ายที่ชัดเจน และสร้างวินัยการเงินแยกกระเป๋าส่วนตัวกับธุรกิจ",
            "maintain": "หาความรู้เพิ่มเติมด้านการจัดการและการวางแผนงบประมาณ"
        },
        2: { # Master Leader
            "strength": "ท่านมีความพร้อมรอบด้าน เป็นที่ต้องการของแหล่งเงินทุน",
            "urgent": "พิจารณาขยายธุรกิจหรือลงทุนในนวัตกรรมเพื่อสร้างความได้เปรียบระยะยาว",
            "maintain": "รักษามาตรฐานระบบการจัดการและเทคโนโลยีให้ทันสมัยอยู่เสมอ"
        }
    }
    
    rec = recs.get(cluster_id, recs)
    
    st.success(f"✅ **จุดแข็งที่ควรรักษา:** {rec['strength']}")
    st.error(f"⚠️ **สิ่งที่ต้องทำด่วน:** {rec['urgent']}")
    st.info(f"🛡️ **ข้อแนะนำเพิ่มเติม:** {rec['maintain']}")
    
    st.markdown("---")
    if st.button("ถัดไป: โปรไฟล์และการบริจาค >"):
        navigate_to('profile')

# --- หน้าที่ 6: Profile & Donation ---
def show_profile():
    st.markdown("## 👤 โปรไฟล์และการกุศล")
    st.write("เพื่อให้งานวิจัยนี้สมบูรณ์ โปรดระบุข้อมูลเพื่อการอ้างอิงและการบริจาค")
    
    with st.form("profile_form"):
        name = st.text_input("ชื่อ-นามสกุล (ระบุหรือไม่ก็ได้)")
        email = st.text_input("อีเมล (เพื่อรับผลประเมิน)")
        
        st.markdown("**💌 ผู้วิจัยจะบริจาคเงิน 100 บาท ให้กับองค์กรที่ท่านเลือก:**")
        charity = st.radio("เลือกองค์กรการกุศล:", 
            ["สภากาชาดไทย", "มูลนิธิรามาธิบดี", "มูลนิธิชัยพัฒนา", "มูลนิธิกระจกเงา"])
            
        submitted = st.form_submit_button("ยืนยันและส่งข้อมูล")
        
        if submitted:
            st.balloons()
            st.success("ขอบพระคุณที่ร่วมเป็นส่วนหนึ่งของงานวิจัย!")
            st.markdown(f"""
            <div style='background-color:#e8f5e9;padding:20px;border-radius:10px;text-align:center;'>
                <h3>🙏 ขอบคุณครับ</h3>
                <p>ข้อมูลของท่าน <b>{name}</b> ได้ถูกบันทึกแล้ว<br>
                และผู้วิจัยจะดำเนินการบริจาคให้ <b>{charity}</b> ต่อไป</p>
            </div>
            """, unsafe_allow_html=True)
            
            # ลิงก์ไปหน้า Support หรือจบการทำงาน
            if st.button("กลับสู่หน้าแรก"):
                navigate_to('landing')

# ==========================================
# 5. Main App Logic
# ==========================================
if st.session_state.page == 'landing':
    show_landing()
elif st.session_state.page == 'input_step1':
    show_input_step1()
elif st.session_state.page == 'input_step2':
    show_input_step2()
elif st.session_state.page == 'dashboard':
    show_dashboard()
elif st.session_state.page == 'recommendation':
    show_recommendation()
elif st.session_state.page == 'profile':
    show_profile()
