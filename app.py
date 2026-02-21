import streamlit as st
import pandas as pd
import numpy as np
import joblib
import zipfile
import os
import plotly.graph_objects as go
import streamlit.components.v1 as components
from autogluon.tabular import TabularPredictor

# ==========================================
# 1. ตั้งค่าหน้าเว็บและธีม (NOMOS Style)
# ==========================================
st.set_page_config(
    page_title="SME FinCheck",
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

# --- ฟังก์ชันสั่งเลื่อนหน้าจอขึ้นบนสุด (อัปเดตแก้ปัญหาหน้า 4 ไม่เลื่อน) ---
def scroll_to_top():
    js = """
        <script>
            // หน่วงเวลา 0.15 วินาที (150ms) เพื่อรอให้ Streamlit วาดหน้าจอให้เสร็จก่อน
            setTimeout(function() {
                // 1. สั่งเลื่อนหน้าต่างหลัก (Window)
                window.parent.scrollTo(0, 0);
                
                // 2. สั่งเลื่อนกรอบเนื้อหาของ Streamlit (ครอบคลุมทั้งคอมพิวเตอร์และมือถือ)
                var containers = window.parent.document.querySelectorAll('.main, .block-container, .stApp');
                for (var i = 0; i < containers.length; i++) {
                    containers[i].scrollTop = 0;
                }
            }, 150); 
        </script>
    """
    components.html(js, height=0)

# ==========================================
# 4. ส่วนแสดงผล (Page Views)
# ==========================================

# --- หน้าที่ 1: Landing Page (แก้ไข: SME FinCheck เป็น Jost Light สีชมพู + ปุ่มมีกรอบเทา) ---
def show_landing():
    # 1. ฝัง CSS (Sarabun + Jost + Button Styling)
    st.markdown("""
        <style>
        /* นำเข้า Font: Sarabun (ไทย) และ Jost (อังกฤษ) */
        @import url('https://fonts.googleapis.com/css2?family=Jost:wght@300;400;600&family=Sarabun:wght@300;400;600&display=swap');
        
        /* บังคับใช้ฟอนต์ Sarabun กับทุกส่วนของเว็บเป็นหลัก */
        html, body, [class*="css"], h1, h2, h3, button, input, select, label, div {
            font-family: 'Sarabun', sans-serif !important;
        }
        
        /* ปรับหัวข้อสีน้ำเงินเข้ม */
        h1, h2, h3 { color: #1E3A8A !important; font-weight: 600; }
        
        /* --- ส่วนตกแต่งปุ่มกด (Start) --- */
        /* 1. สถานะปกติ (มีกรอบเทาตลอดเวลา) */
        div[data-testid="stBaseButton-primary"] > button, 
        button[kind="primary"] {
            transition: all 0.3s ease !important;
            border-radius: 8px !important;
            border: 2px solid #A9A9A9 !important; /* กรอบสีเทา */
        }
        
        /* 2. สถานะเมื่อเอาเมาส์ไปชี้ (Hover) -> เปลี่ยนพื้นหลังเป็นสีชมพู */
        div[data-testid="stBaseButton-primary"] > button:hover,
        button[kind="primary"]:hover {
            background-color: #FE5C8D !important;  /* สีชมพูจุฬาฯ */
            border-color: #A9A9A9 !important;     /* กรอบยังคงเป็นสีเทา */
            color: white !important;              /* ตัวอักษรสีขาว */
            box-shadow: 0 4px 15px rgba(254, 92, 141, 0.4) !important;
            transform: scale(1.05) !important;
        }

        /* Hero Text */
        .hero-title {
            font-family: 'Sarabun', sans-serif !important;
            font-size: 2.5em !important;
            font-weight: bold;
            color: #1E3A8A;
            text-align: center;
            margin-top: 20px;
            margin-bottom: 10px;
            line-height: 1.3;
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

    # 3. ข้อความ Hero Text (แก้ไขตามที่ขอ)
    # บรรทัดบน: สีน้ำเงินเข้ม ฟอนต์ Sarabun (ตาม Class เดิม)
    # บรรทัดล่าง: SME FinCheck เป็นสีชมพูจุฬาฯ (#FE5C8D) และฟอนต์ Jost แบบบาง (Weight 500)
    st.markdown("""
        <div class="hero-title">
            ตรวจสุขภาพธุรกิจและการเงินด้วย<br>
            <span style='font-family: "Jost", sans-serif; font-weight: 500; color: #FE5C8D; font-size: 1.1em;'>SME FinCheck</span>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="hero-subtitle">รู้ทันสุขภาพการเงิน | ประเมิน DNA ธุรกิจ | ลดความเสี่ยง | รับคำแนะนำ</div>', unsafe_allow_html=True)

    st.markdown("---")

    # 4. ปุ่มกด Start
    c_btn1, c_btn2, c_btn3 = st.columns([1, 2, 1]) 
    with c_btn2:
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
    scroll_to_top() # <--- ใส่ไว้บรรทัดแรก
    
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
    st.markdown("<h3 style='font-family: Sarabun; font-weight: 600; color: #1E3A8A;'>🧬 DNA ธุรกิจของท่าน</h3>", unsafe_allow_html=True)
    
    st.info(
        "💡 **กรุณาประเมินระดับการดำเนินงาน**\n\n"
        "**0** = ไม่มี &nbsp;&nbsp;•&nbsp;&nbsp; "
        "**1** = น้อยที่สุด &nbsp;&nbsp;•&nbsp;&nbsp; "
        "**5** = มากที่สุด"
    )

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
    scroll_to_top() 

    # 1. ฝัง CSS
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Sarabun:wght@300;400;600&display=swap');
        html, body, [class*="css"], h1, h2, h3, h4, h5, h6, p, span, div, label, button, input, select {
            font-family: 'Sarabun', sans-serif !important;
        }
        .stButton>button {
            background-color: #FE5C8D !important;
            color: white !important;
            border: 1px solid #ddd !important;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown('<p style="color: #888; font-size: 1.1em; margin-bottom: 0;">ขั้นตอนที่ 2/2: ระดับดำเนินงาน</p>', unsafe_allow_html=True)
    st.markdown("<h3 style='color: #1E3A8A; margin-top: 0;'>💼 ระดับดำเนินงาน</h3>", unsafe_allow_html=True)

    st.info("💡 **กรุณาประเมินระดับการดำเนินงาน**\n\n**0** = ไม่มี   •   **1** = น้อยที่สุด   •   **5** = มากที่สุด")

    # ✅ ใช้คำสั่ง list(range(6)) เพื่อสร้างเลข 0 ถึง 5 อัตโนมัติ (แก้ปัญหาระบบแชทกลืนตัวเลข)
    score_options = list(range(6))
    binary_options = ["ไม่มี (0)", "มี (1)"]

    with st.form("form_step2"):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("<p style='color: #1E3A8A; font-weight: bold;'>ผู้ประกอบการและทีมงาน</p>", unsafe_allow_html=True)
            cap_netw = st.selectbox("ท่านใช้เครือข่ายหรือพันธมิตรในการดำเนินธุรกิจในระดับใด", score_options, index=0)
            csr3 = st.selectbox("กิจการของท่านมีระบบกำจัดของเสีย", binary_options, index=0)
            ohr_career = st.selectbox("กิจการของท่านมีเส้นทางอาชีพให้พนักงานรับรู้", binary_options, index=0)
        
        with col2:
            st.markdown("<p style='color: #1E3A8A; font-weight: bold;'>การบัญชีและสถานการณ์เศรษฐกิจ</p>", unsafe_allow_html=True)
            prc_cfw = st.selectbox("กระแสเงินสดเพื่อประกอบธุรกิจและชำระหนี้อยู่ในระดับใด", score_options, index=0)
            eco_adt = st.selectbox("กิจการของท่านสามารถในการปรับตัวรับสถานการณ์เศรษฐกิจในระดับใด", score_options, index=0)
        
        with col3:
            st.markdown("<p style='color: #1E3A8A; font-weight: bold;'>เทคโนโลยีและการสื่อสาร</p>", unsafe_allow_html=True)
            ecm_net = st.selectbox("การเข้าถึงเครือข่ายอินเตอร์เน็ตของกิจการอยู่ในระดับใด", score_options, index=0)
            res_ch = st.selectbox("ความสามารถในการโต้ตอบลูกค้าผ่านช่องทางต่าง ๆ อยู่ในระดับใด", score_options, index=0)

        st.markdown("---")
        submitted = st.form_submit_button("🚀 ประเมินผลลัพธ์", type="primary", use_container_width=True)

        if submitted:
            csr3_val = 1 if "มี" in csr3 else 0
            ohr_career_val = 1 if "มี" in ohr_career else 0
            
            st.session_state.inputs.update({
                'CAP_NETW': cap_netw, 'CSR3': csr3_val, 'OHR_CAREER': ohr_career_val,
                'PRC_CFW': prc_cfw, 'ECO_ADT': eco_adt,
                'ECM_NET': ecm_net, 'RES_CH': res_ch,
            })
            
            success = process_results()
            if success:
                navigate_to('dashboard') 

# --- ฟังก์ชันประมวลผล (Processing Logic) ---
def process_results():
    prob = 0.5
    cluster_id = 0
    inputs = st.session_state.inputs

    # 1. Clustering Logic (DNA ธุรกิจ)
    cluster_features = ['BEH_MON', 'BRN_IMAGE', 'BRN_BRAND', 'SAV_VIRUS', 'SAV_PDPA', 'CRI_PLN', 'POL_BEN', 'POL_ADJ']
    cluster_vals = [inputs.get(f, 0) for f in cluster_features]
    
    try:
        X_cluster = pd.DataFrame([cluster_vals], columns=cluster_features)
        X_scaled = scaler_model.transform(X_cluster)
        raw_cluster_id = kmeans_model.predict(X_scaled)
        cluster_id = int(np.ravel(raw_cluster_id).item())
    except Exception as e:
        print(f"Cluster Error: {e}")
        cluster_id = 0
        
    st.session_state.results['cluster_id'] = cluster_id

    # 2. Prediction Logic (AutoGluon)
    if predictor_model is not None and not df_raw.empty:
        try:
            pred_df = df_raw.head(1).copy().reset_index(drop=True)
            
            for col in pred_df.columns:
                pred_df.at[0, col] = float('nan')
            
            for key, val in inputs.items():
                if key in pred_df.columns:
                    pred_df.at[0, key] = float(val) if isinstance(val, (int, float)) else val

            if 'SIZ' in pred_df.columns: pred_df.at[0, 'SIZ'] = 1 
            if 'YER' in pred_df.columns: pred_df.at[0, 'YER'] = 10 
                
            prob_df = predictor_model.predict_proba(pred_df)
            
            # ✅ แก้ปัญหา _iLocIndexer ด้วยการดึงค่าผ่าน numpy array (flatten) รับรองไม่พัง
            prob_array = prob_df.values.flatten()
            prob = float(prob_array[-1]) 
            
            # =======================================================
            # 🌟 ส่วนเทียบบัญญัติไตรยางศ์ (Min-Max Scaling จากค่าทดสอบจริง)
            # =======================================================
            MAX_RAW_PROB = 0.491  # เพดานความเสี่ยงสูงสุด (เมื่อตอบ 0 หมด)
            MIN_RAW_PROB = 0.099  # พื้นความเสี่ยงต่ำสุด (เมื่อตอบ 5 และมีครบหมด)
            
            # สมการเทียบบัญญัติไตรยางศ์: (ค่าปัจจุบัน - ค่าต่ำสุด) / (ค่าสูงสุด - ค่าต่ำสุด)
            # ป้องกันกรณีตัวหารเป็น 0 (เผื่อไว้)
            if MAX_RAW_PROB > MIN_RAW_PROB:
                scaled_prob = (prob - MIN_RAW_PROB) / (MAX_RAW_PROB - MIN_RAW_PROB)
            else:
                scaled_prob = 0
                
            risk_score = scaled_prob * 100
            
            # ดักไว้ไม่ให้คะแนนทะลุ 100 หรือติดลบ (ในกรณีที่โมเดลสวิงเกินค่าที่เทสไว้เล็กน้อย)
            risk_score = min(100.0, max(0.0, risk_score))
            # =======================================================

        except Exception as e:
            st.error(f"🚨 ข้อผิดพลาดจากระบบพยากรณ์: {e}")
            return False 
    else:
        # กรณีไม่มีโมเดล (Fallback)
        score = inputs.get('PRC_CFW', 0) * 0.4 + inputs.get('CAP_NETW', 0) * 0.3 + inputs.get('BEH_MON', 0) * 0.3
        prob = 1 - (score / 5.0)
        risk_score = prob * 100

    # บันทึกผลลัพธ์ลงระบบ
    st.session_state.results['risk_prob'] = prob          # เก็บค่าดิบจาก AutoGluon ไว้ (เผื่อใช้งานในอนาคต)
    st.session_state.results['risk_score'] = risk_score   # ค่าสเกล 0-100% ที่แปลงแล้วสำหรับโชว์กราฟ
    
    return True

# --- หน้าที่ 4: Dashboard (Result) - ฉบับแก้ไข Syntax Error (วงเล็บครบ) ---
def show_dashboard():
    scroll_to_top() # <--- ใส่ไว้บรรทัดแรก
    
    # 1. ฝัง CSS (เพิ่มส่วนตกแต่งปุ่มกด)
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Sarabun:wght@300;400;600&display=swap');
        
        html, body, [class*="css"], h1, h2, h3, button, input, select, label, div {
            font-family: 'Sarabun', sans-serif !important;
        }
        h1, h2, h3 { color: #1E3A8A !important; font-weight: 600; }

        /* --- ปรับแต่งปุ่มกดแบบ Primary (ปุ่ม Recommendation) --- */
        /* สถานะปกติ: พื้นขาว กรอบเทา */
        div[data-testid="stBaseButton-primary"] > button,
        button[kind="primary"] {
            background-color: white !important;
            color: #333 !important;                 
            border: 2px solid #A9A9A9 !important;   
            border-radius: 8px !important;
            transition: all 0.3s ease !important;
        }

        /* สถานะ Hover: พื้นชมพูจุฬา ตัวอักษรขาว */
        div[data-testid="stBaseButton-primary"] > button:hover,
        button[kind="primary"]:hover {
            background-color: #FF5C8D !important;   /* สีชมพู Chula */
            border-color: #FF5C8D !important;       /* กรอบสีชมพู */
            color: white !important;                
            box-shadow: 0 4px 10px rgba(255, 92, 141, 0.4) !important;
            transform: scale(1.02) !important;
        }
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
    # 2. ดึงผลลัพธ์จาก AI (แก้ไข: ลบสูตรเดิมทิ้งทั้งหมด)
    # ==========================================
    cluster_id = st.session_state.results.get('cluster_id', 1)
    risk_score = st.session_state.results.get('risk_score', 50.0)
    
    # กรณี cluster_id คืนค่ามาเป็น Array จาก K-Means (เช่น ) ให้ดึงตัวเลขออกมา
    if isinstance(cluster_id, (np.ndarray, list)):
        cluster_id = int(cluster_id)

    # ==========================================
    # 3. ส่วนแสดงผล (Display)
    # ==========================================
    
    # Mapping DNA
    cluster_info = {
        0: {"name": "Active Marketer (นักการตลาดไฟแรง)", "color": "#F9D607", # เหลือง (กลาง)
            "desc": "โดดเด่นด้านการตลาดและภาพลักษณ์องค์กร ควรเสริมสร้างระบบเทคโนโลยีและการบริหารความเสี่ยงหลังบ้าน"},
        1: {"name": "Potential Starter (นักสู้ผู้มีศักยภาพ)", "color": "#e74c3c", # แดง (สูง)
            "desc": "มีความยืดหยุ่น ควรสร้างวินัยทางการเงินและวางระบบบัญชีให้น่าเชื่อถือ เพื่อเพิ่มโอกาสเข้าถึงแหล่งเงินทุน"},
        2: {"name": "Master Leader (ผู้นำระดับมาสเตอร์)", "color": "#2ecc71", # เขียว (ต่ำ)
            "desc": "ความพร้อมรอบด้าน ทั้งด้านการเงิน การตลาด และการรับมือวิกฤตการณ์ ธนาคารและนักลงทุนพร้อมสนับสนุนแหล่งเงินทุน"}
    }
    
    # ดึงค่า DNA (ใช้ .get เพื่อป้องกัน Error)
    dna = cluster_info.get(cluster_id, cluster_info[1])

    st.markdown(f"<h3 style='text-align:center; color:#1E3A8A;'>📊 ผลการประเมินสุขภาพการเงิน</h3>", unsafe_allow_html=True)
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🧬 DNA ธุรกิจของคุณ", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="background-color: {dna['color']}; padding: 20px; border-radius: 10px; color: white; text-align: center; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
            <h3 style='margin:0; font-family: Sarabun, sans-serif; color: white !important; text-shadow: 1px 1px 2px rgba(0,0,0,0.3);'>{dna['name']}</h3>
            <p style='margin-top:10px; font-size: 1.1em; font-family: Sarabun, sans-serif;'>{dna['desc']}</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.write("")
        st.markdown("#### 💡 คำแนะนำเบื้องต้น:", unsafe_allow_html=True)
        
        # Logic คำแนะนำ
        if cluster_id == 1: # Potential (แดง)
            st.warning("⚠️ ควรเร่งจัดทำบัญชีรายรับ-รายจ่ายให้ชัดเจน และลดภาระหนี้ที่ไม่จำเป็น")
        elif cluster_id == 0: # Active (เหลือง)
            st.info("ℹ️ การตลาดยอดเยี่ยม เข้าใจผู้บริโภค แต่ต้องอุดรูรั่วความปลอดภัยของระบบ IT")
        else: # Master (เขียว)
            st.success("✅ เครดิตดี เตรียมเอกสารยื่นกู้เพื่อขยายกิจการได้เลย")

    with col2:
        st.markdown(f"### 🔮 มีข้อจำกัดการเข้าถึงแหล่งเงินทุน: **{risk_score:.1f}%**", unsafe_allow_html=True)
        
        # ==========================================
        # 🌟 โค้ดกราฟ Gauge Chart (แสดงข้อความ สูง/ปานกลาง/ต่ำ)
        # ==========================================
        
        # 1. กำหนดเงื่อนไขคำที่จะแสดงตรงกลางกราฟ ตามช่วงคะแนน
        if risk_score < 40:
            risk_level_text = "ต่ำ"
            text_color = "#1b5e20"  # เขียวเข้ม
        elif risk_score <= 70:
            risk_level_text = "ปานกลาง"
            text_color = "#b8860b"  # เหลืองทอง
        else:
            risk_level_text = "สูง"
            text_color = "#842029"  # แดงเข้ม

        # 2. สร้างกราฟ Gauge (ซ่อนตัวเลขตรงกลาง)
        fig = go.Figure(go.Indicator(
            mode = "gauge",  
            value = risk_score,
            gauge = {
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "gray", 'tickvals': [0, 40, 70, 100]},
                'bar': {'color': "darkblue"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 40], 'color': "#2ecc71"},   # เขียว 
                    {'range': [40, 70], 'color': "#F9D607"},  # เหลือง 
                    {'range': [70, 100], 'color': "#e74c3c"}  # แดง 
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': risk_score
                }
            }
        ))

        # 3. แปะข้อความ สูง/ปานกลาง/ต่ำ ลงไปตรงกลางเกจ
        fig.add_annotation(
            x=0.5, y=0.10,  
            text=f"<b>{risk_level_text}</b>",
            font=dict(size=60, color=text_color, family="Sarabun"),
            showarrow=False
        )
        
        fig.update_layout(height=300, margin=dict(l=20, r=20, t=30, b=20), font={'family': "Sarabun"})
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    
    c_btn1, c_btn2, c_btn3 = st.columns([0.15, 0.7, 0.15])
    with c_btn2:
        if st.button("📄 ดูข้อเสนอแนะโดยละเอียด (Recommendation)", type="primary", use_container_width=True):
            navigate_to('recommendation')

# --- หน้าที่ 5: Recommendations (ปรับแต่งขนาดตัวอักษรและไอคอน + ผสาน AI 2 ตัว) ---
def show_recommendation():
    scroll_to_top() # <--- ใส่ไว้บรรทัดแรก

    # 1. ฝัง CSS (Sarabun + ปุ่ม Hover ชมพูจุฬาฯ)
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Sarabun:wght@300;400;600&display=swap');
        html, body, [class*="css"], h1, h2, h3, button, input, select, label, div {
            font-family: 'Sarabun', sans-serif !important;
        }
        
        /* --- ปรับแต่งปุ่มกดแบบ Primary --- */
        /* สถานะปกติ: พื้นขาว กรอบเทา */
        div[data-testid="stBaseButton-primary"] > button,
        button[kind="primary"] {
            background-color: white !important;
            color: #333 !important;                 
            border: 2px solid #A9A9A9 !important;   
            border-radius: 8px !important;
            transition: all 0.3s ease !important;
        }

        /* สถานะ Hover: พื้นชมพูจุฬา ตัวอักษรขาว */
        div[data-testid="stBaseButton-primary"] > button:hover,
        button[kind="primary"]:hover {
            background-color: #FF5C8D !important;
            border-color: #FF5C8D !important;
            color: white !important;
            box-shadow: 0 4px 10px rgba(255, 92, 141, 0.4) !important;
            transform: scale(1.02) !important;
        }
        </style>
    """, unsafe_allow_html=True)

    # 2. หัวข้อหลัก (สีน้ำเงิน #1E3A8A)
    st.markdown("<h3 style='color: #1E3A8A;'>🎯 คำแนะนำสำหรับท่าน (Recommendations)</h3>", unsafe_allow_html=True)
    st.markdown("---")

    # ดึงค่าผลลัพธ์จาก AI ทั้ง 2 ตัว
    if 'results' not in st.session_state:
        st.session_state.results = {'cluster_id': 0, 'risk_score': 50.0}
    
    cluster_id = st.session_state.results.get('cluster_id', 0)
    risk_score = st.session_state.results.get('risk_score', 50.0)

    # ✅ แก้ Error: แปลง cluster_id จาก Array ให้เป็นตัวเลข Integer ธรรมดา
    if isinstance(cluster_id, (np.ndarray, list)):
        cluster_id = int(cluster_id)
    else:
        cluster_id = int(cluster_id)

    # ---------------------------------------------------------
    # ส่วนที่ 1: คำแนะนำด้านการเงิน ยึดตามความเสี่ยง AutoGluon (Risk Score)
    # ---------------------------------------------------------
    if risk_score > 70:
        urgent_advice = "ควรเร่งสร้างวินัยทางการเงิน จัดทำบัญชีรายรับ-รายจ่ายให้ชัดเจน และลดภาระหนี้ที่ไม่จำเป็นด่วน ธนาคาร นักลงทุนและเจ้าหนี้พิจารณา 'กระแสเงินสด' ที่น่าเชื่อถือเป็นหลัก"
    elif risk_score >= 41:
        urgent_advice = "กิจการของท่านยังพอประคองตัวได้ แต่ควรระวังการใช้เงินเกินตัว ควรเริ่มจัดเตรียมเอกสารทางการเงินให้เป็นระบบ จัดเตรียมพร้อมด้านไอทีและการรองรับวิกฤติการณ์ต่าง ๆ ที่อาจเกิดขึ้น"
    else:
        urgent_advice = "กิจการมีเครือข่ายธุรกิจที่ดี บัญชีและกระแสเงินสดน่าเชื่อถือทำให้เครดิตอยู่ในเกณฑ์ยอดเยี่ยม ระบบไอทีมีความพร้อม สามารถปรับตัวกับเศรษฐกิจและวิกฤติการณ์ได้ เตรียมแผนธุรกิจเพื่อยื่นขอเงินทุนขยายกิจการได้เลย"

    # ---------------------------------------------------------
    # ส่วนที่ 2: คำแนะนำด้านการจัดการ ยึดตาม DNA ธุรกิจ (K-Means)
    # ---------------------------------------------------------
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

    rec = recs.get(cluster_id, recs)

    # --- แสดงผลหน้าจอ (ปรับตาม Format สีสันสวยงามที่ท่านออกแบบไว้) ---
    
    # 1. กล่องผลลัพธ์ (พื้นหลังสีเทาอ่อน #F2F2F2)
    st.markdown(f"""
        <div style='background-color: #F2F2F2; padding: 15px; border-radius: 8px; border: 1px solid #ddd; margin-bottom: 25px;'>
        <p style='color: #1E3A8A; font-size: 1.1em; margin-bottom: 5px;'><b>💼 ผลลัพธ์ (จากข้อจำกัดการเข้าถึงแหล่งเงินทุน {risk_score:.1f}%)</b></p>
        <p style='margin-bottom: 0;'>{urgent_advice}</p>
        </div>
    """, unsafe_allow_html=True)

    # 2. แบ่ง 3 คอลัมน์ สำหรับ จุดเด่น, อัปเกรดด่วน, รักษาไว้
    col_rec1, col_rec2, col_rec3 = st.columns(3)

    # คอลัมน์ 1: จุดเด่น (พื้นหลังสีเขียวอ่อน #E2EFD9)
    with col_rec1:
        st.markdown(f"""
            <div style='background-color: #E2EFD9; padding: 20px; border-radius: 10px; height: 100%; box-shadow: 0 2px 4px rgba(0,0,0,0.05);'>
            <p style='color: #2e7d32; font-size: 1.1em; margin-bottom: 10px;'><b>✅ จุดเด่น:</b></p>
            <p style='font-size: 0.95em; line-height: 1.5;'>{rec['strength']}</p>
            </div>
        """, unsafe_allow_html=True)

    # คอลัมน์ 2: อัปเกรดด่วน (พื้นหลังสีเหลือง/ส้มอ่อน #FFF2CC)
    with col_rec2:
        st.markdown(f"""
            <div style='background-color: #FFF2CC; padding: 20px; border-radius: 10px; height: 100%; box-shadow: 0 2px 4px rgba(0,0,0,0.05);'>
            <p style='color: #c62828; font-size: 1.1em; margin-bottom: 10px;'><b>🚀 อัปเกรดด่วน:</b></p>
            <p style='font-size: 0.95em; line-height: 1.5;'>{rec['urgent']}</p>
            </div>
        """, unsafe_allow_html=True)

    # คอลัมน์ 3: รักษาไว้ (พื้นหลังสีฟ้าอ่อน #DEEAF6)
    with col_rec3:
        st.markdown(f"""
            <div style='background-color: #DEEAF6; padding: 20px; border-radius: 10px; height: 100%; box-shadow: 0 2px 4px rgba(0,0,0,0.05);'>
            <p style='color: #1565c0; font-size: 1.1em; margin-bottom: 10px;'><b>🛡️ รักษาไว้:</b></p>
            <p style='font-size: 0.95em; line-height: 1.5;'>{rec['maintain']}</p>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True) # เพิ่มช่องว่างด้านล่างเล็กน้อยให้อ่านสบายตา
    st.markdown("---")

    # ปุ่มกดไปหน้า Profile
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("ถัดไป: โปรไฟล์ >", type="primary", use_container_width=True):
            navigate_to('profile')

# --- หน้าที่ 6: Profile & Survey (TAM) - ฉบับแก้ไขข้อความและปุ่ม ---
def show_profile():
    scroll_to_top() # <--- ใส่ไว้บรรทัดแรก
    
    # 1. ฝัง CSS (Sarabun + ปุ่มกด/ปุ่มลิงก์ Hover ชมพูจุฬาฯ)
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Sarabun:wght@300;400;600&display=swap');
        
        html, body, [class*="css"], h1, h2, h3, h4, h5, button, input, select, label, div, p, a {
            font-family: 'Sarabun', sans-serif !important;
        }

        /* --- 1. ปรับแต่งปุ่มกดแบบ Primary (ปุ่ม "ยืนยัน" ใน Form) --- */
        div[data-testid="stFormSubmitButton"] > button,
        button[kind="primary"] {
            background-color: white !important;
            color: #333 !important;                 
            border: 2px solid #A9A9A9 !important;   
            border-radius: 8px !important;
            transition: all 0.3s ease !important;
        }
        /* Hover: สีชมพูจุฬา ตัวอักษรขาว */
        div[data-testid="stFormSubmitButton"] > button:hover,
        button[kind="primary"]:hover {
            background-color: #FF5C8D !important;
            border-color: #FF5C8D !important;
            color: white !important;                
            box-shadow: 0 4px 10px rgba(255, 92, 141, 0.4) !important;
            transform: scale(1.02) !important;
        }

        /* --- 2. ปรับแต่งปุ่มลิงก์ (Link Button ทำแบบสอบถาม) --- */
        div[data-testid="stLinkButton"] > a {
            background-color: white !important;
            color: #333 !important;
            border: 2px solid #A9A9A9 !important;
            border-radius: 8px !important;
            transition: all 0.3s ease !important;
            text-decoration: none !important;
        }
        /* Link Button Hover: สีชมพูจุฬา ตัวอักษรขาว */
        div[data-testid="stLinkButton"] > a:hover {
            background-color: #FF5C8D !important;
            border-color: #FF5C8D !important;
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
        
        # --- ข้อความสีชมพูจุฬาฯ ก่อนปุ่มยืนยัน (จัดกึ่งกลาง) ---
        st.markdown("<p style='color:#FF5C8D; font-weight:bold; text-align:center;'>โปรดกดยืนยันเพื่อตอบแบบสอบถามในลำดับถัดไปครับ</p>", unsafe_allow_html=True)
        
        # สร้างคอลัมน์เพื่อบีบขนาดปุ่มให้อยู่ตรงกลาง (สัดส่วน 1 : 2 : 1)
        col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
        with col_btn2:
            submitted = st.form_submit_button("ยืนยัน", type="primary", use_container_width=True)
        
    if submitted:
        st.balloons() # ลูกโป่งลอย
        st.success("ขอบพระคุณที่ร่วมเป็นส่วนหนึ่งของงานวิจัย!")
        
        # กล่องขอบคุณ
        st.markdown(f"""
        <div style='background-color:#e8f5e9; padding:20px; border-radius:10px; text-align:center; border: 1px solid #c8e6c9; margin-bottom: 20px;'>
            <h3 style='color:#2e7d32; margin-bottom:10px;'>🙏 ขอความกรุณากดลิงค์เพื่อตอบแบบสอบถามด้านล่าง</h3>
            <p style='font-size: 1.1em; color:#1b5e20;'>
                ข้อมูลของท่าน <b>{name if name else ''}</b> ได้ถูกบันทึกแล้ว<br>
                ขอบคุณครับ
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # --- ปุ่มลิงก์ไป MS Forms (แก้ไขข้อความตามที่ขอ) ---
        ms_form_url = "https://forms.office.com/r/yr6x0jdH3T"  # <--- 🔴 อย่าลืมใส่ลิงก์ MS Forms ของคุณตรงนี้นะครับ
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            # ข้อความปุ่มเปลี่ยนเป็น: "ทำแบบสอบถามแสดงความเห็นต่อเครื่องมือที่ทดลองใช้"
            # CSS ด้านบนจะทำให้ปุ่มนี้ Hover แล้วเป็นสีชมพู
            st.link_button("📝 ทำแบบสอบถามแสดงความเห็นต่อ SME FinCheck", ms_form_url, use_container_width=True)

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
