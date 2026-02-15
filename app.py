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

# Custom CSS: หัวข้อ = Sarabun, เนื้อหา = Sarabun
st.markdown("""
<style>
    /* 1. นำเข้า Font จาก Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Sarabun:wght@300;400;600;700&family=Sarabun:wght@300;400;500;700&display=swap');
    
    /* 2. กำหนด Font พื้นฐาน (เนื้อหา) เป็น Sarabun */
    html, body, [class*="css"], p, div, label, .stMarkdown, .stTextInput, .stNumberInput, .stSelectbox {
        font-family: 'Sarabun', sans-serif;
        color: #333333;
    }

    /* 3. กำหนด Font หัวข้อ (Header) เป็น Sarabun */
    h1, h2, h3, h4, h5, h6, .stTitle {
        font-family: 'Sarabun', sans-serif !important;
        font-weight: 600; /* ปรับความหนาให้ดูเด่น */
    }
    
    /* 4. ปรับแต่งปุ่มกด (Button) ให้เป็น Sarabun เพื่อความสวยงาม */
    .stButton>button {
        font-family: 'Sarabun', sans-serif !important;
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
        font-family: 'Sarabun', sans-serif;
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
        font-family: 'Sarabun', sans-serif;
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

# --- หน้าที่ 1: Landing Page (แก้ไข: กรอบเทาตลอดเวลา + Hover สีชมพู) ---
def show_landing():
    # 1. ฝัง CSS (Sarabun + Button Styling)
    st.markdown("""
        <style>
        /* นำเข้า Font Sarabun */
        @import url('https://fonts.googleapis.com/css2?family=Sarabun:wght@300;400;600&display=swap');
        
        /* บังคับใช้ฟอนต์ Sarabun กับทุกส่วนของเว็บ */
        html, body, [class*="css"], h1, h2, h3, button, input, select, label, div {
            font-family: 'Sarabun', sans-serif !important;
        }
        
        /* ปรับหัวข้อสีน้ำเงินเข้ม */
        h1, h2, h3 { color: #1E3A8A !important; font-weight: 600; }
        
        /* --- ส่วนที่เพิ่มใหม่: ตกแต่งปุ่มกด (Start) --- */
        /* 1. สถานะปกติ (มีกรอบเทาตลอดเวลา) */
        div[data-testid="stBaseButton-primary"] > button, 
        button[kind="primary"] {
            transition: all 0.3s ease !important;
            border-radius: 8px !important;
            
            /* --- จุดที่แก้ไข: ใส่กรอบสีเทาไว้ตรงนี้เลย เพื่อให้โชว์ตลอดเวลา --- */
            border: 2px solid #A9A9A9 !important; 
            /* ----------------------------------------------------------- */
        }
        
        /* 2. สถานะเมื่อเอาเมาส์ไปชี้ (Hover) -> เปลี่ยนพื้นหลังเป็นสีชมพู */
        div[data-testid="stBaseButton-primary"] > button:hover,
        button[kind="primary"]:hover {
            background-color: #FF5C8D !important;  /* พื้นหลังเปลี่ยนเป็นสีชมพูจุฬา */
            border-color: #A9A9A9 !important;     /* กรอบยังคงเป็นสีเทาเหมือนเดิม */
            color: white !important;              /* ตัวอักษรสีขาว */
            box-shadow: 0 4px 15px rgba(255, 92, 141, 0.4) !important;
            transform: scale(1.05) !important;
        }
        /* ---------------------------------- */

        /* Hero Text */
        .hero-title {
            font-family: 'Sarabun', sans-serif !important;
            font-size: 2.5em !important;
            font-weight: bold;
            color: #1E3A8A;
            text-align: center;
            margin-top: 20px;
            margin-bottom: 10px;
        }
        .hero-subtitle {
            font-family: 'Sarabun', sans-serif !important;
            font-size: 1.2em !important;
            color: #555;
            text-align: center;
            margin-bottom: 30px;
        }
        </style>
    """, unsafe_allow_html=True)

    # 2. แสดงรูปภาพ (บีบให้เหลือ 50% ของหน้าจอ)
    c_img1, c_img2, c_img3 = st.columns([1, 2, 1]) 
    with c_img2:
        try:
            st.image("FinCheck.jpg", use_container_width=True) 
        except:
            st.error("ไม่พบไฟล์รูปภาพ (FinCheck.jpg)")

    # 3. ข้อความ Hero Text
    st.markdown('<div class="hero-title">ตรวจเช็คแหล่งเงินทุน<br>ของคุณในเสี้ยวนาที</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-subtitle">รู้ทันสุขภาพการเงิน | ประเมิน DNA ธุรกิจ | ลดความเสี่ยง | รับคำแนะนำ</div>', unsafe_allow_html=True)

    st.markdown("---")

    # 4. ปุ่มกด Start
    c_btn1, c_btn2, c_btn3 = st.columns([1, 2, 1]) 
    with c_btn2:
        # ปุ่มนี้จะได้รับผลจาก CSS ด้านบน
        if st.button("🚀 เริ่มประเมินทันที (Start)", type="primary", use_container_width=True):
            navigate_to('input_step1')

    st.markdown("---")

    # Footer
    st.markdown("""
    <div style='text-align: center; color: #888; font-size: 0.9em; margin-top: 20px;'>
        พัฒนาโดย: <b>นายสมเกียรติ จูสวัสดิ์</b><br>
        นิสิตปริญญาเอก | หลักสูตรธุรกิจเทคโนโลยีและการจัดการนวัตกรรม<br>
        จุฬาลงกรณ์มหาวิทยาลัย
    </div>
    """, unsafe_allow_html=True)
        
# --- หน้าที่ 2: Input Step 1 (DNA) ---
def show_input_step1():
    # 1. ฝัง CSS (Sarabun + สีหัวข้อ)
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Sarabun:wght@300;400;600&display=swap');
        
        html, body, [class*="css"], h1, h2, h3, h4, h5, button, input, select, label, div {
            font-family: 'Sarabun', sans-serif !important;
        }
        
        /* ปรับสีหัวข้อให้เป็นสีน้ำเงิน */
        h1, h2, h3 { color: #1E3A8A !important; font-weight: 600; }
        </style>
    """, unsafe_allow_html=True)
    
    # หัวข้อหลัก
    st.markdown("<h3 style='font-family: Sarabun; font-weight: 600; color: #1E3A8A;'>🧬 DNA ธุรกิจท่าน</h3>", unsafe_allow_html=True)
    
    st.info("💡 โปรดทราบ: 0 = ไม่มี, 1 = น้อยที่สุด, 5 = มากที่สุด")

    # ตัวเลือกคะแนน
    score_options = [0, 1, 2, 3, 4, 5]

    with st.form("form_step1"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("<h5 style='color: #1E3A8A; font-weight: bold;'>การตลาดและผลิตภัณฑ์</h5>", unsafe_allow_html=True)
            beh_mon = st.selectbox("ท่านติดตามและตรวจสอบความพึงพอใจของลูกค้า", score_options, index=0)
            brn_image = st.selectbox("ท่านให้ความสำคัญกับภาพลักษณ์องค์กร", score_options, index=0)
            brn_brand = st.selectbox("การรับรู้และความน่าเชื่อถือของแบรนด์ของท่าน", score_options, index=0)
        
        with col2:
            st.markdown("<h5 style='color: #1E3A8A; font-weight: bold;'>เทคโนโลยีและการรับมือสถานการณ์</h5>", unsafe_allow_html=True)
            sav_virus = st.selectbox("การอัพเดทโปรแกรมป้องกันไวรัสเพื่อความปลอดภัยของระบบงาน", score_options, index=0)
            sav_pdpa = st.selectbox("ท่านปฏิบัติตามกฎหมาย PDPA เพื่อรักษาข้อมูลลูกค้า", score_options, index=0)
            cri_pln = st.selectbox("ท่านมีแผนรองรับวิกฤตการณ์ต่าง ๆ เช่น ภัยสงคราม โรคระบาด แผ่นดินไหว เป็นต้น", score_options, index=0)

        with col3:
            st.markdown("<h5 style='color: #1E3A8A; font-weight: bold;'>นโยบายภาครัฐ</h5>", unsafe_allow_html=True)
            pol_ben = st.selectbox("ท่านได้รับประโยชน์จากนโยบายภาครัฐ", score_options, index=0)
            pol_adj = st.selectbox("ท่านสามารถปรับรูปแบบธุรกิจให้สอดคล้องนโยบายรัฐ", score_options, index=0)

        st.markdown("---")
        
        # ปุ่ม Submit (สีแดงมาตรฐาน)
        submitted = st.form_submit_button("ถัดไป >", type="primary", use_container_width=True)
        
        if submitted:
            if 'inputs' not in st.session_state:
                st.session_state.inputs = {}

            st.session_state.inputs.update({
                'BEH_MON': beh_mon, 'BRN_IMAGE': brn_image, 'BRN_BRAND': brn_brand,
                'SAV_VIRUS': sav_virus, 'SAV_PDPA': sav_pdpa, 'CRI_PLN': cri_pln,
                'POL_BEN': pol_ben, 'POL_ADJ': pol_adj
            })
            navigate_to('input_step2')

# --- หน้าที่ 3: Input Step 2 (Business Mgmt) ---
def show_input_step2():
    # 1. ฝัง CSS (Sarabun + สีหัวข้อ)
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Sarabun:wght@300;400;600&display=swap');
        
        html, body, [class*="css"], h1, h2, h3, h4, h5, button, input, select, label, div {
            font-family: 'Sarabun', sans-serif !important;
        }
        
        /* ปรับแต่งปุ่มกดให้เหมือนหน้าที่แล้ว (ถ้าต้องการ) */
        button[kind="primary"] {
             background-color: white !important;
             color: #333 !important;
             border: 2px solid #A9A9A9 !important;
        }
        button[kind="primary"]:hover {
             background-color: #FF5C8D !important;
             border-color: #A9A9A9 !important;
             color: white !important;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="step-indicator">ขั้นตอนที่ 2/2: ระดับดำเนินงาน</div>', unsafe_allow_html=True)
    
    # แก้ไข 1: หัวข้อหลักเป็นสีน้ำเงิน (#1E3A8A)
    st.markdown("<h3 style='font-family: Sarabun, sans-serif; font-weight: 600; color: #1E3A8A;'>💼 ระดับดำเนินงาน</h3>", unsafe_allow_html=True)
    
    st.info("💡 โปรดเลือกคำตอบ: 0 = ไม่มี, 1 = น้อยที่สุด, 5 = มากที่สุด")

    # --- กำหนดตัวเลือกสำหรับ Dropdown ---
    score_options = [0, 1, 2, 3, 4, 5]
    binary_options = ["ไม่มี (0)", "มี (1)"]

    with st.form("form_step2"):
        # --- แบ่งเป็น 3 คอลัมน์ ---
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # แก้ไข 2: หัวข้อย่อยเป็นสีน้ำเงิน
            st.markdown("<h5 style='color: #1E3A8A; font-weight: bold;'>ผู้ประกอบการและทีมงาน</h5>", unsafe_allow_html=True)
            cap_netw = st.selectbox("ท่านใช้เครือข่ายหรือพันธมิตรในการดำเนินธุรกิจในระดับใด", score_options, index=0)
            # ใช้ Dropdown แบบ มี/ไม่มี
            csr3 = st.selectbox("กิจการของท่านมีระบบกำจัดของเสีย", binary_options, index=0)
            ohr_career = st.selectbox("กิจการของท่านมีเส้นทางอาชีพให้พนักงานรับรู้", binary_options, index=0)
            
        with col2:
            # แก้ไข 3: หัวข้อย่อยเป็นสีน้ำเงิน
            st.markdown("<h5 style='color: #1E3A8A; font-weight: bold;'>การบัญชีและสถานการณ์เศรษฐกิจ</h5>", unsafe_allow_html=True)
            prc_cfw = st.selectbox("กระแสเงินสดเพื่อประกอบธุรกิจและชำระหนี้อยู่ในระดับใด", score_options, index=0)
            eco_adt = st.selectbox("กิจการของท่านสามารถในการปรับตัวรับสถานการณ์เศรษฐกิจในระดับใด (ECO_ADT)", score_options, index=0)            

        with col3:                  
            # แก้ไข 4: หัวข้อย่อยเป็นสีน้ำเงิน
            st.markdown("<h5 style='color: #1E3A8A; font-weight: bold;'>เทคโนโลยีและการสื่อสาร</h5>", unsafe_allow_html=True)
            ecm_net = st.selectbox("การเข้าถึงเครือข่ายอินเตอร์เน็ตของกิจการอยู่ในระดับใด", score_options, index=0)
            res_ch = st.selectbox("ความสามารถในการโต้ตอบลูกค้าผ่านช่องทางต่าง ๆ อยู่ในระดับใด", score_options, index=0)

        st.markdown("---")

        # --- ส่วนปุ่มกด ---
        col_b1, col_b2 = st.columns(2)
        
        with col_b1:
            # ปุ่มย้อนกลับ
            if st.form_submit_button("< ย้อนกลับ", type="secondary", use_container_width=True):
                navigate_to('input_step1')
                
        with col_b2:
            # ปุ่มประเมินผลลัพธ์
            submitted = st.form_submit_button("🚀 ประเมินผลลัพธ์", type="primary", use_container_width=True)
            
        if submitted:
            # แปลงค่า
            csr3_val = 1 if "มี" in csr3 else 0
            ohr_career_val = 1 if "มี" in ohr_career else 0

            # บันทึกค่าลง Session
            st.session_state.inputs.update({
                'CAP_NETW': cap_netw, 'CSR3': csr3_val, 'OHR_CAREER': ohr_career_val,
                'PRC_CFW': prc_cfw, 'ECO_ADT': eco_adt,
                'ECM_NET': ecm_net, 'RES_CH': res_ch,
            })
            
            # ไปหน้า Dashboard
            navigate_to('dashboard') # ไปหน้าแสดงผล

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

# --- หน้าที่ 4: Dashboard (Result) - แก้ไข Logic ความเสี่ยงให้ถูกต้อง ---
def show_dashboard():
    # 1. ฝัง CSS (Sarabun + สีปุ่ม)
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Sarabun:wght@300;400;600&display=swap');
        html, body, [class*="css"], h1, h2, h3, button, input, select, label, div {
            font-family: 'Sarabun', sans-serif !important;
        }
        h1, h2, h3 { color: #1E3A8A !important; font-weight: 600; }
        </style>
    """, unsafe_allow_html=True)

    # ตรวจสอบข้อมูล
    if 'inputs' not in st.session_state or not st.session_state.inputs:
        st.warning("⚠️ กรุณากรอกข้อมูลในขั้นตอนที่ 1 และ 2 ให้ครบถ้วนก่อนครับ")
        if st.button("กลับไปกรอกข้อมูล"):
            navigate_to('input_step1')
        return

    inputs = st.session_state.inputs

    # ==========================================
    # 2. ส่วนประมวลผล (Calculation Logic)
    # ==========================================
    
    # --- 2.1 คำนวณ DNA (Clustering) ---
    cluster_features = ['BEH_MON', 'BRN_IMAGE', 'BRN_BRAND', 'SAV_VIRUS', 'SAV_PDPA', 'CRI_PLN', 'POL_BEN', 'POL_ADJ']
    try:
        cluster_vals = [inputs.get(f, 0) for f in cluster_features]
        if 'scaler_model' in globals() and 'kmeans_model' in globals():
            X_cluster = pd.DataFrame([cluster_vals], columns=cluster_features)
            X_scaled = scaler_model.transform(X_cluster)
            cluster_id = int(kmeans_model.predict(X_scaled))
        else:
            cluster_id = 0
    except:
        cluster_id = 0

    # --- 2.2 คำนวณความเสี่ยง (Risk Prediction) ---
    # แก้ไข: ดึงค่า Probability ของ Class 1 (ความเสี่ยง) ให้ถูกต้อง
    try:
        if 'predictor_model' in globals() and predictor_model is not None:
            # เตรียมข้อมูล
            if 'df_raw' in globals() and not df_raw.empty:
                pred_df = df_raw.iloc[0:1].copy().reset_index(drop=True)
                for c in df_raw.columns:
                     if c not in inputs and c not in ['ID', 'target']:
                        if df_raw[c].dtype == 'object': pred_df[c] = df_raw[c].mode()
                        else: pred_df[c] = df_raw[c].mean()
            else:
                 pred_df = pd.DataFrame([inputs])
                 
            for k, v in inputs.items():
                if k in pred_df.columns: pred_df[k] = v
            
            if 'SIZ' not in pred_df: pred_df['SIZ'] = 1
            if 'YER' not in pred_df: pred_df['YER'] = 10

            # ✅ จุดที่แก้ไข: ดึงค่าความน่าจะเป็นของ Class 1 (เสี่ยง/มีข้อจำกัด)
            # ใช้ [2] เพื่อระบุคอลัมน์ Class 1 และ .iloc เพื่อดึงค่า scalar
            proba_df = predictor_model.predict_proba(pred_df)
            if 1 in proba_df.columns:
                prob = proba_df[2].iloc
            else:
                prob = proba_df.iloc # กันเหนียว (แถว 0, คอลัมน์ 1)
                
        else:
            raise Exception("No Model")
    except:
        # Fallback Logic (คำนวณมือ กรณี Model Error)
        score_sum = inputs.get('PRC_CFW', 0)*0.4 + inputs.get('CAP_NETW', 0)*0.3 + inputs.get('BEH_MON', 0)*0.3
        # คะแนนน้อย = ความเสี่ยงสูง (1 - คะแนน/5)
        prob = 1 - (score_sum / 5.0)
        prob = max(0.1, min(0.9, prob))

    # แปลงเป็นเปอร์เซ็นต์ (0-100)
    risk_score = prob * 100
    
    # บันทึกผล
    st.session_state.results['cluster_id'] = cluster_id
    st.session_state.results['risk_score'] = risk_score

    # ==========================================
    # 3. ส่วนแสดงผล (Display)
    # ==========================================
    
    cluster_info = {
        0: {"name": "Active Marketer (นักการตลาดไฟแรง)", "color": "#f39c12", 
            "desc": "โดดเด่นด้านการตลาดและภาพลักษณ์องค์กร ควรเสริมสร้างระบบเทคโนโลยีและการบริหารความเสี่ยงหลังบ้าน"},
        1: {"name": "Potential Starter (นักสู้ผู้มีศักยภาพ)", "color": "#e74c3c", 
            "desc": "มีความยืดหยุ่น ควรสร้างวินัยทางการเงินและวางระบบบัญชีให้น่าเชื่อถือ เพื่อเพิ่มโอกาสเข้าถึงแหล่งเงินทุน"},
        2: {"name": "Master Leader (ผู้นำระดับมาสเตอร์)", "color": "#2ecc71", 
            "desc": "ความพร้อมรอบด้าน ทั้งด้านการเงิน การตลาด และการรับมือวิกฤตการณ์ ธนาคารและนักลงทุนพร้อมสนับสนุนแหล่งเงินทุน"}
    }
    dna = cluster_info.get(cluster_id, cluster_info)

    st.markdown(f"<h3 style='text-align:center; color:#1E3A8A;'>📊 ผลการประเมินสุขภาพการเงิน</h3>", unsafe_allow_html=True)
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🧬 DNA ธุรกิจของคุณ", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="background-color: {dna['color']}; padding: 20px; border-radius: 10px; color: white; text-align: center; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
            <h3 style='margin:0; font-family: Sarabun, sans-serif; color: white !important;'>{dna['name']}</h3>
            <p style='margin-top:10px; font-size: 1.1em; font-family: Sarabun, sans-serif;'>{dna['desc']}</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.write("")
        st.markdown("#### 💡 คำแนะนำเบื้องต้น:", unsafe_allow_html=True)
        if cluster_id == 1:
            st.warning("⚠️ **ความเสี่ยงสูง:** ควรเร่งจัดทำบัญชีรายรับ-รายจ่ายให้ชัดเจน และลดภาระหนี้ที่ไม่จำเป็น")
        elif cluster_id == 0:
            st.info("ℹ️ **พอใช้:** การตลาดยอดเยี่ยม เข้าใจผู้บริโภค แต่ต้องอุดรูรั่วความปลอดภัยของระบบ IT, PDPA")
        else:
            st.success("✅ **ยอดเยี่ยม:** เครดิตดี เตรียมเอกสารยื่นกู้เพื่อขยายกิจการได้เลย")

    with col2:
        st.markdown(f"### 🔮 ความเสี่ยงการเข้าถึงแหล่งเงิน: **{risk_score:.1f}%**", unsafe_allow_html=True)
        
        # กราฟ Gauge Chart (แก้ไข: สีแดงคือเสี่ยงสูง 71-100)
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = risk_score,
            gauge = {
                'axis': {'range': , 'tickwidth': 1, 'tickcolor': "gray"},
                'bar': {'color': "darkblue"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 40], 'color': "#2ecc71"},   # เขียว (0-40) = เสี่ยงต่ำ
                    {'range': [41, 70], 'color': "#f1c40f"},  # เหลือง (40-70) = ปานกลาง
                    {'range': [71, 100], 'color': "#e74c3c"}  # แดง (70-100) = เสี่ยงสูง
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': risk_score
                }
            }
        ))
        
        fig.update_layout(height=300, margin=dict(l=20, r=20, t=30, b=20), font={'family': "Sarabun"})
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    
    # ปุ่มไปหน้า Recommendation
    c_btn1, c_btn2, c_btn3 = st.columns([0.15, 0.7, 0.15])
    with c_btn2:
        if st.button("📄 ดูข้อเสนอแนะโดยละเอียด (Recommendation)", type="primary", use_container_width=True):
            navigate_to('recommendation')

# --- หน้าที่ 5: Recommendations (ปรับแต่งขนาดตัวอักษรและไอคอน) ---
def show_recommendation():
    # 1. ฝัง CSS (Sarabun + ปุ่ม Hover ชมพู)
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Sarabun:wght@300;400;600&display=swap');
        
        html, body, [class*="css"], h1, h2, h3, h4, h5, button, input, select, label, div {
            font-family: 'Sarabun', sans-serif !important;
        }

        /* --- ปรับแต่งปุ่มกด (Next Button) --- */
        div[data-testid="stBaseButton-primary"] > button,
        button[kind="primary"] {
            background-color: white !important;
            color: #333 !important;                 
            border: 2px solid #A9A9A9 !important;   
            border-radius: 8px !important;
            transition: all 0.3s ease !important;
        }

        div[data-testid="stBaseButton-primary"] > button:hover,
        button[kind="primary"]:hover {
            background-color: #FF5C8D !important;   /* สีชมพู Chula */
            border-color: #A9A9A9 !important;       /* กรอบสีเทาเหมือนเดิม */
            color: white !important;                
            box-shadow: 0 4px 10px rgba(255, 92, 141, 0.4) !important;
            transform: scale(1.02) !important;
        }
        </style>
    """, unsafe_allow_html=True)

    # 2. หัวข้อหลัก (สีน้ำเงิน #1E3A8A)
    st.markdown("<h3 style='color:#1E3A8A; font-weight:bold;'>🎯 คำแนะนำสำหรับท่าน (Recommendations)</h3>", unsafe_allow_html=True)
    st.markdown("---")

    # ดึงค่า cluster_id
    if 'results' not in st.session_state:
        st.session_state.results = {'cluster_id': 0}
    
    cluster_id = st.session_state.results.get('cluster_id', 0)
    
    # Recommendation Logic
    recs = {
        0: { # Active Marketer
            "strength": "กิจการของท่านมีความเข้มแข็งด้านการตลาด การสร้างแบรนด์และภาพลักษณ์องค์กร",
            "urgent": "ควรสร้างความปลอดภัยทางเทคโนโลยี รักษาข้อมูลส่วนบุคคลของลูกค้า และกำหนดแผนรองรับวิกฤตการณ์ด่วน! ธนาคารและนักลงทุนมองว่านี่คือความเสี่ยงแฝง",
            "maintain": "รักษาฐานลูกค้าเอาไว้ให้มั่น และเสริมสร้างการตลาดออนไลน์ให้ต่อเนื่อง"
        },
        1: { # Potential Starter
            "strength": "กิจการของท่านมีความยืดหยุ่นและมีโอกาสในการเริ่มต้นวางระบบองค์กรที่ถูกต้อง",
            "urgent": "ควรเริ่มจัดทำบัญชีรายรับ-รายจ่ายที่ชัดเจน น่าเชื่อถือ และเสริมสร้างวินัยการเงิน แยกกระเป๋าส่วนตัวออกจากกระเป๋าของธุรกิจ ธนาคารและนักลงทุนต้องการตัวเลขที่น่าเชื่อถือ",
            "maintain": "ยึดมั่นความตั้งใจธุรกิจเอาไว้ หาความรู้เพิ่มเติมด้านการจัดการ และการวางแผนงบประมาณ"
        },
        2: { # Master Leader
            "strength": "กิจการของท่านมีความพร้อมรอบด้าน ธนาคารและนักลงทุนพอใจกับกิจการลักษณะนี้",
            "urgent": "ควรหาโอกาสขยายธุรกิจให้เติบโตยิ่งขึ้น ลงทุนในนวัตกรรมเพื่อสร้างความได้เปรียบระยะยาว",
            "maintain": "รักษามาตรฐานระบบการจัดการ ส่งเสริมการตลาดและผลิตภัณฑ์ และเทคโนโลยีให้ทันสมัยอยู่เสมอ"
        }
    }
    
    rec = recs.get(cluster_id, recs[0])
    
    # --- แสดงผลแบบ HTML เพื่อปรับขนาดตัวอักษรและไอคอน ---
    
    # 1. จุดแข็ง (สีเขียว)
    st.markdown(f"""
        <div style="background-color: #d1e7dd; padding: 15px; border-radius: 8px; border: 1px solid #badbcc; margin-bottom: 15px;">
            <h4 style="color: #0f5132; margin: 0; font-family: Sarabun; font-weight: bold;">✅ จุดแข็งที่ควรรักษา:</h4>
            <div style="color: #0f5132; margin-top: 8px; font-size: 1.1rem; font-family: Sarabun;">
                {rec['strength']}
            </div>
        </div>
    """, unsafe_allow_html=True)

    # 2. สิ่งที่ต้องทำด่วน (สีแดง)
    st.markdown(f"""
        <div style="background-color: #f8d7da; padding: 15px; border-radius: 8px; border: 1px solid #f5c6cb; margin-bottom: 15px;">
            <h4 style="color: #842029; margin: 0; font-family: Sarabun; font-weight: bold;">⚠️ สิ่งที่ต้องทำด่วน:</h4>
            <div style="color: #842029; margin-top: 8px; font-size: 1.1rem; font-family: Sarabun;">
                {rec['urgent']}
            </div>
        </div>
    """, unsafe_allow_html=True)

    # 3. ข้อแนะนำเพิ่มเติม (สีฟ้า)
    st.markdown(f"""
        <div style="background-color: #cff4fc; padding: 15px; border-radius: 8px; border: 1px solid #b6effb; margin-bottom: 15px;">
            <h4 style="color: #055160; margin: 0; font-family: Sarabun; font-weight: bold;">🛡️ ข้อแนะนำเพิ่มเติม:</h4>
            <div style="color: #055160; margin-top: 8px; font-size: 1.1rem; font-family: Sarabun;">
                {rec['maintain']}
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # ปุ่มกดไปหน้า Profile
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("ถัดไป: โปรไฟล์ >", type="primary", use_container_width=True):
            navigate_to('profile')

# --- หน้าที่ 6: Profile & Survey (TAM) - ฉบับแก้ไขข้อความและปุ่ม ---
def show_profile():
    # 1. ฝัง CSS (Sarabun + ปุ่ม Hover ชมพู + ปุ่ม Link)
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Sarabun:wght@300;400;600&display=swap');
        
        html, body, [class*="css"], h1, h2, h3, h4, h5, button, input, select, label, div, p, a {
            font-family: 'Sarabun', sans-serif !important;
        }

        /* --- ปรับแต่งปุ่มกดทั่วไป (ปุ่ม "ยืนยัน" ใน Form) --- */
        div[data-testid="stForm"] button[kind="secondary"] {
            background-color: white !important;
            color: #333 !important;                 
            border: 2px solid #A9A9A9 !important;   
            border-radius: 8px !important;
            transition: all 0.3s ease !important;
        }
        /* Hover: สีชมพูจุฬา ตัวอักษรขาว */
        div[data-testid="stForm"] button[kind="secondary"]:hover {
            background-color: #FF5C8D !important;   
            border-color: #A9A9A9 !important;
            color: white !important;                
            box-shadow: 0 4px 10px rgba(255, 92, 141, 0.4) !important;
            transform: scale(1.02) !important;
        }

        /* --- ปรับแต่งปุ่มลิงก์ (Link Button ไป MS Forms) --- */
        a[data-testid="stLinkButton"] {
            background-color: white !important;
            color: #333 !important;
            border: 2px solid #A9A9A9 !important;
            border-radius: 8px !important;
            text-align: center !important;
            text-decoration: none !important;
            transition: all 0.3s ease !important;
            display: inline-flex;
            justify-content: center;
            align-items: center;
        }
        
        /* Link Button Hover: สีชมพูจุฬา ตัวอักษรขาว */
        a[data-testid="stLinkButton"]:hover {
            background-color: #FF5C8D !important;
            border-color: #A9A9A9 !important;
            color: white !important;
            box-shadow: 0 4px 10px rgba(255, 92, 141, 0.4) !important;
            transform: scale(1.02) !important;
        }
        </style>
    """, unsafe_allow_html=True)

    # 2. หัวข้อหลัก (สีน้ำเงิน #1E3A8A)
    st.markdown("<h2 style='color:#1E3A8A; font-weight:bold;'>👤 โปรไฟล์</h2>", unsafe_allow_html=True)
    st.write("เพื่อให้งานวิจัยนี้สมบูรณ์ โปรดบันทึกข้อมูลเพื่อการอ้างอิง")
    
    with st.form("profile_form"):
        name = st.text_input("ชื่อ-นามสกุล (ระบุหรือไม่ก็ได้)")
        email = st.text_input("อีเมล (เพื่อรับผลประเมินในภายหลัง)")
        
        st.write("") # เว้นบรรทัด
        
        # --- ข้อความสีชมพูจุฬาฯ ก่อนปุ่มยืนยัน ---
        st.markdown("<p style='color:#FF5C8D; font-weight:bold;'>โปรดกดยืนยันเพื่อตอบแบบสอบถามในลำดับถัดไปครับ</p>", unsafe_allow_html=True)
        
        # ปุ่มยืนยัน (CSS จะทำให้ Hover เป็นสีชมพู)
        submitted = st.form_submit_button("ยืนยัน")
        
    if submitted:
        st.balloons() # ลูกโป่งลอย
        st.success("ขอบพระคุณที่ร่วมเป็นส่วนหนึ่งของงานวิจัย!")
        
        # กล่องขอบคุณ
        st.markdown(f"""
        <div style='background-color:#e8f5e9; padding:20px; border-radius:10px; text-align:center; border: 1px solid #c8e6c9; margin-bottom: 20px;'>
            <h3 style='color:#2e7d32; margin-bottom:10px;'>🙏 ขอความกรุณาตอบแบบสอบถามด้านล่าง</h3>
            <p style='font-size: 1.1em; color:#1b5e20;'>
                ข้อมูลของท่าน <b>{name if name else ''}</b> ได้ถูกบันทึกแล้ว<br>
                ขอบคุณครับ
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # --- ปุ่มลิงก์ไป MS Forms (แก้ไขข้อความตามที่ขอ) ---
        ms_form_url = "https://forms.office.com/r/YOUR_FORM_ID"  # <--- 🔴 อย่าลืมใส่ลิงก์ MS Forms ของคุณตรงนี้นะครับ
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            # ข้อความปุ่มเปลี่ยนเป็น: "ทำแบบสอบถามแสดงความเห็นต่อเครื่องมือที่ทดลองใช้"
            # CSS ด้านบนจะทำให้ปุ่มนี้ Hover แล้วเป็นสีชมพู
            st.link_button("📝 ทำแบบสอบถามแสดงความเห็นต่อเครื่องมือที่ทดลองใช้", ms_form_url, use_container_width=True)

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
