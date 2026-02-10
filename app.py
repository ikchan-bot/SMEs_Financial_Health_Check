import streamlit as st
import pandas as pd
import numpy as np
import joblib
import zipfile
import os
import plotly.graph_objects as go
from autogluon.tabular import TabularPredictor

# 1. ตั้งค่าหน้าเว็บ
st.set_page_config(page_title="SME Health Check", page_icon="📊", layout="wide")
st.markdown("""<style>.metric-card {background-color: #f0f2f6; border-radius: 10px; padding: 20px; box-shadow: 2px 2px 10px rgba(0,0,0,0.1);} h1 { color: #0e1117; } h3 { color: #262730; }</style>""", unsafe_allow_html=True)

st.title("📊 ระบบประเมินสุขภาพธุรกิจและโอกาสเข้าถึงเงินทุน SMEs")
st.markdown("---")

# 2. ฟังก์ชันโหลดโมเดล (ตัด st.toast ออกแล้ว เพื่อแก้ Error)
@st.cache_resource
def load_resources():
    # 2.1 โหลด Clustering
    kmeans = joblib.load('kmeans_behavior_model.joblib')
    scaler = joblib.load('scaler_behavior.joblib')
    
    # 2.2 โหลด AutoGluon (ระบบรวมร่าง)
    extract_path = './autogluon_model_extracted'
    combined_zip_name = 'full_model_combined.zip'
    
    if not os.path.exists(extract_path):
        # รวมไฟล์
        part_files = sorted([f for f in os.listdir('.') if f.startswith('model_part_')])
        if not part_files:
            raise FileNotFoundError("ไม่พบไฟล์ model_part_*.zip")
            
        with open(combined_zip_name, 'wb') as combined_file:
            for part in part_files:
                with open(part, 'rb') as p:
                    combined_file.write(p.read())
        
        # แตกไฟล์
        with zipfile.ZipFile(combined_zip_name, 'r') as zip_ref:
            zip_ref.extractall(extract_path)

    # ค้นหาไฟล์ predictor.pkl
    model_path = extract_path
    found = False
    for root, dirs, files in os.walk(extract_path):
        if 'predictor.pkl' in files:
            model_path = root
            found = True
            break
            
    if not found:
        raise FileNotFoundError("หาไฟล์ predictor.pkl ไม่เจอ")

    # โหลดโมเดล (ปิดการเช็คเวอร์ชัน)
    predictor = TabularPredictor.load(model_path, require_py_version_match=False)
    
    # 2.3 โหลดข้อมูลดิบ
    df_raw = pd.read_excel('RawData2.xlsx')
    
    return kmeans, scaler, predictor, df_raw

try:
    with st.spinner('กำลังประมวลผลและโหลด AI... (ครั้งแรกจะนานหน่อยนะครับ)'):
        kmeans_model, scaler_model, predictor_model, df_raw = load_resources()
    st.success("✅ ระบบพร้อมใช้งาน!") # ย้ายมาไว้ข้างนอกแทน
except Exception as e:
    st.error(f"❌ เกิดข้อผิดพลาด: {e}")
    st.stop()

# 3. ส่วนรับข้อมูล
st.sidebar.header("📝 กรอกข้อมูลกิจการ")
with st.sidebar.form("input_form"):
    st.subheader("1. พฤติกรรม (1-5)")
    c1, c2 = st.columns(2)
    with c1:
        beh_mon = st.slider("วินัยการเงิน", 1, 5, 3)
        brn_brand = st.slider("สร้างแบรนด์", 1, 5, 3)
        sav_virus = st.slider("ป้องกันโรค", 1, 5, 3)
        pol_ben = st.slider("สวัสดิการ", 1, 5, 3)
    with c2:
        brn_image = st.slider("ภาพลักษณ์", 1, 5, 3)
        sav_pdpa = st.slider("มาตรการ PDPA", 1, 5, 1)
        cri_pln = st.slider("แผนรับมือวิกฤต", 1, 5, 2)
        pol_adj = st.slider("ปรับตัวนโยบาย", 1, 5, 3)
        
    st.subheader("2. การเงิน")
    prc_cfw = st.number_input("สัดส่วนกระแสเงินสด", 0.0, 10.0, 0.5)
    cap_netw = st.number_input("มูลค่าส่วนทุน (บาท)", 0, 100000000, 1000000)
    yer = st.number_input("อายุธุรกิจ (ปี)", 0, 100, 5)
    
    submitted = st.form_submit_button("🚀 วิเคราะห์ผล")

# 4. ประมวลผล
if submitted:
    # Prepare Clustering
    features = ['BEH_MON', 'BRN_IMAGE', 'BRN_BRAND', 'SAV_VIRUS', 'SAV_PDPA', 'CRI_PLN', 'POL_BEN', 'POL_ADJ']
    vals = [beh_mon, brn_image, brn_brand, sav_virus, sav_pdpa, cri_pln, pol_ben, pol_adj]
    cluster_id = kmeans_model.predict(scaler_model.transform(pd.DataFrame([vals], columns=features)))[0]

    # Prepare Prediction
    pred_df = df_raw.iloc[0:1].copy().reset_index(drop=True)
    for c in df_raw.columns:
        if c not in ['ID', 'target']:
            if df_raw[c].dtype == 'object': pred_df[c] = df_raw[c].mode()[0]
            else: pred_df[c] = df_raw[c].mean()
            
    pred_df['BEH_MON'] = beh_mon; pred_df['SAV_PDPA'] = sav_pdpa; pred_df['PRC_CFW'] = prc_cfw
    pred_df['CAP_NETW'] = cap_netw; pred_df['YER'] = yer

    prob = predictor_model.predict_proba(pred_df).iloc[0, 1]
    
    # Display
    c_color = ["#f1c40f", "#e74c3c", "#2ecc71"] # Yellow, Red, Green based on ID 0,1,2
    c_names = ["The Traditional Marketer", "The Vulnerable", "The Resilient Leader"]
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"### 🧬 DNA: <span style='color:{c_color[cluster_id]}'>{c_names[cluster_id]}</span>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"### 🔮 ความเสี่ยง: **{prob*100:.1f}%**")
        fig = go.Figure(go.Indicator(mode="gauge+number", value=prob*100, domain={'x': [0, 1], 'y': [0, 1]},
            gauge={'axis': {'range': [None, 100]}, 'bar': {'color': "darkblue"}, 'steps': [{'range': [0, 50], 'color': "#a3e4d7"}, {'range': [50, 100], 'color': "#fadbd8"}]}))
        fig.update_layout(height=250, margin=dict(l=20,r=20,t=20,b=20))
        st.plotly_chart(fig, use_container_width=True)
        
    st.success("วิเคราะห์ข้อมูลเรียบร้อย!")
