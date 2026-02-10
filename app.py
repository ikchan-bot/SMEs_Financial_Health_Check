import streamlit as st
import pandas as pd
import numpy as np
import joblib
import zipfile
import os
import plotly.graph_objects as go
from autogluon.tabular import TabularPredictor

# ==========================================
# 1. ตั้งค่าหน้าเว็บ (UI Setup)
# ==========================================
st.set_page_config(
    page_title="SME Financial & Strategic Health Check",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
    }
    h1 { color: #0e1117; }
    h3 { color: #262730; }
</style>
""", unsafe_allow_html=True)

st.title("📊 ระบบประเมินสุขภาพธุรกิจและโอกาสเข้าถึงเงินทุน SMEs")
st.markdown("---")

# ==========================================
# 2. ฟังก์ชันโหลดโมเดล (Load Brains)
# ==========================================
@st.cache_resource
def load_resources():
    # 2.1 โหลด Clustering Model
    kmeans = joblib.load('kmeans_behavior_model.joblib')
    scaler = joblib.load('scaler_behavior.joblib')
    
    # 2.2 ระบบประกอบร่างไฟล์ Zip และค้นหาโมเดล (Auto-Finder)
    extract_path = './autogluon_model_extracted'
    combined_zip_name = 'full_model_combined.zip'
    
    # ถ้ายังไม่มีโฟลเดอร์ ให้ทำการแตกไฟล์
    if not os.path.exists(extract_path):
        st.toast("กำลังประกอบร่างโมเดล AI...", icon="🧩")
        
        # รวมไฟล์ย่อย
        part_files = sorted([f for f in os.listdir('.') if f.startswith('model_part_')])
        if not part_files:
            raise FileNotFoundError("ไม่พบไฟล์ model_part_*.zip กรุณาอัปโหลดไฟล์ย่อยให้ครบ")

        with open(combined_zip_name, 'wb') as combined_file:
            for part in part_files:
                with open(part, 'rb') as p:
                    combined_file.write(p.read())
        
        # แตกไฟล์
        with zipfile.ZipFile(combined_zip_name, 'r') as zip_ref:
            zip_ref.extractall(extract_path)

    # เดินหาไฟล์ predictor.pkl ไม่ว่าจะซ่อนอยู่ลึกแค่ไหน
    model_path = extract_path
    found = False
    for root, dirs, files in os.walk(extract_path):
        if 'predictor.pkl' in files:
            model_path = root
            found = True
            break
            
    if not found:
        raise FileNotFoundError(f"หาไฟล์ predictor.pkl ไม่เจอใน {extract_path}")

    # --- จุดที่แก้: เพิ่ม require_py_version_match=False ---
    predictor = TabularPredictor.load(model_path, require_py_version_match=False)
    
    # 2.3 โหลดข้อมูลดิบ
    df_raw = pd.read_excel('RawData2.xlsx')
    
    return kmeans, scaler, predictor, df_raw

try:
    with st.spinner('กำลังโหลดสมอง AI... (อาจใช้เวลา 1-2 นาทีในครั้งแรก)'):
        kmeans_model, scaler_model, predictor_model, df_raw = load_resources()
    st.toast("✅ ระบบพร้อมใช้งาน!", icon="🚀")
except Exception as e:
    st.error(f"❌ เกิดข้อผิดพลาด: {e}")
    st.stop()

# ==========================================
# 3. ส่วนรับข้อมูล (Sidebar Inputs)
# ==========================================
st.sidebar.header("📝 กรอกข้อมูลกิจการ")

with st.sidebar.form("input_form"):
    st.subheader("1. พฤติกรรมและการจัดการ (1-5)")
    col_sb1, col_sb2 = st.columns(2)
    with col_sb1:
        beh_mon = st.slider("วินัยการเงิน", 1, 5, 3, help="ความสม่ำเสมอในการทำบัญชี/เดินบัญชี")
        brn_brand = st.slider("สร้างแบรนด์", 1, 5, 3)
        sav_virus = st.slider("ป้องกันโรค", 1, 5, 3)
        pol_ben = st.slider("สวัสดิการ", 1, 5, 3)
    with col_sb2:
        brn_image = st.slider("ภาพลักษณ์", 1, 5, 3)
        sav_pdpa = st.slider("มาตรการ PDPA", 1, 5, 1, help="การคุ้มครองข้อมูลส่วนบุคคล")
        cri_pln = st.slider("แผนรับมือวิกฤต", 1, 5, 2)
        pol_adj = st.slider("ปรับตัวนโยบาย", 1, 5, 3)
        
    st.markdown("---")
    st.subheader("2. ข้อมูลทางการเงิน")
    prc_cfw = st.number_input("สัดส่วนกระแสเงินสด (Cash Flow Ratio)", value=0.5, step=0.1)
    cap_netw = st.number_input("มูลค่าส่วนทุน (Net Worth - บาท)", value=1_000_000, step=100_000)
    yer = st.number_input("อายุธุรกิจ (ปี)", value=5, step=1)
    
    submitted = st.form_submit_button("🚀 วิเคราะห์ผล (Analyze)")

# ==========================================
# 4. การประมวลผลและแสดงผล
# ==========================================
if submitted:
    # --- 4.1 เตรียมข้อมูล Clustering ---
    cluster_features = ['BEH_MON', 'BRN_IMAGE', 'BRN_BRAND', 'SAV_VIRUS', 
                        'SAV_PDPA', 'CRI_PLN', 'POL_BEN', 'POL_ADJ']
    input_values = [beh_mon, brn_image, brn_brand, sav_virus, 
                    sav_pdpa, cri_pln, pol_ben, pol_adj]
    
    cluster_df = pd.DataFrame([input_values], columns=cluster_features)
    cluster_scaled = scaler_model.transform(cluster_df)
    cluster_id = kmeans_model.predict(cluster_scaled)[0]

    # Mapping ผลลัพธ์
    if cluster_id == 2:
        c_name = "The Resilient Leader (ผู้นำรอบด้าน)"
        c_desc = "แข็งแกร่งทั้งการเงินและการจัดการความเสี่ยง"
        c_color = "#2ecc71" # Green
    elif cluster_id == 0:
        c_name = "The Traditional Marketer (เน้นภาพลักษณ์)"
        c_desc = "โดดเด่นเรื่องแบรนด์ แต่ขาดมาตรการรองรับความเสี่ยง"
        c_color = "#f1c40f" # Yellow
    else: # Cluster 1
        c_name = "The Vulnerable (กลุ่มเปราะบาง)"
        c_desc = "มีความเสี่ยงเชิงโครงสร้าง ต้องการการยกระดับเร่งด่วน"
        c_color = "#e74c3c" # Red

    # --- 4.2 เตรียมข้อมูล Prediction ---
    # ใช้ค่าเฉลี่ย/ฐานนิยมจากข้อมูลดิบเป็นค่าเริ่มต้น
    pred_df = df_raw.iloc[0:1].copy().reset_index(drop=True)
    for col in df_raw.columns:
        if col not in ['ID', 'target']:
            if df_raw[col].dtype == 'object':
                pred_df[col] = df_raw[col].mode()[0]
            else:
                pred_df[col] = df_raw[col].mean()
    
    # แทนค่าที่ user กรอก
    pred_df['BEH_MON'] = beh_mon
    pred_df['SAV_PDPA'] = sav_pdpa
    pred_df['PRC_CFW'] = prc_cfw
    pred_df['CAP_NETW'] = cap_netw
    pred_df['YER'] = yer

    # พยากรณ์
    prob_urgency = predictor_model.predict_proba(pred_df).iloc[0, 1]
    
    # ==========================================
    # 5. แสดง Dashboard
    # ==========================================
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"### 🧬 DNA ธุรกิจของคุณ: <span style='color:{c_color}'>{c_name}</span>", unsafe_allow_html=True)
        st.info(f"💡 {c_desc}")
        
    with col2:
        risk_percent = prob_urgency * 100
        st.markdown(f"### 🔮 ความเสี่ยงทางการเงิน: **{risk_percent:.1f}%**")
        
        # Gauge Chart
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = risk_percent,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "โอกาสขาดสภาพคล่อง (Financial Urgency)"},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 30], 'color': "#a3e4d7"},
                    {'range': [30, 70], 'color': "#f9e79f"},
                    {'range': [70, 100], 'color': "#fadbd8"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': risk_percent
                }
            }
        ))
        fig_gauge.update_layout(height=250, margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig_gauge, use_container_width=True)

    # --- ส่วนล่าง: Radar & Advice ---
    col3, col4 = st.columns([1, 1])
    
    with col3:
        st.subheader("🕸️ วิเคราะห์จุดแข็ง-จุดอ่อน")
        categories = ['วินัยการเงิน', 'ภาพลักษณ์', 'แบรนด์', 'ป้องกันไวรัส', 
                      'PDPA', 'แผนวิกฤต', 'สวัสดิการ', 'ปรับนโยบาย']
        
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=input_values, theta=categories, fill='toself', name='ธุรกิจของคุณ'
        ))
        fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 5])), height=400)
        st.plotly_chart(fig_radar, use_container_width=True)

    with col4:
        st.subheader("📋 คำแนะนำเชิงกลยุทธ์")
        with st.expander("📌 ข้อแนะนำการบริหาร", expanded=True):
            if cluster_id == 2:
                st.success("✅ **รักษามาตรฐาน:** ท่านทำได้ดีมาก")
            elif cluster_id == 0:
                st.warning("⚠️ **ระวัง:** ต้องเพิ่มมาตรการความปลอดภัยและแผนฉุกเฉิน")
            else:
                st.error("🚨 **ปรับปรุงด่วน:** ต้องสร้างวินัยการเงินและระบบพื้นฐานใหม่")

        with st.expander("💰 ข้อแนะนำการเงิน", expanded=True):
            if prob_urgency > 0.5:
                st.write("🔴 **เสี่ยงสูง:** ควรชะลอการลงทุนและรักษาสภาพคล่อง")
            else:
                st.write("🟢 **ปกติ:** สถานะการเงินแข็งแกร่ง พร้อมขยายธุรกิจ")

else:
    st.info("👈 กรุณากรอกข้อมูลทางด้านซ้ายมือ แล้วกดปุ่ม 'วิเคราะห์ผล' เพื่อเริ่มใช้งาน")
