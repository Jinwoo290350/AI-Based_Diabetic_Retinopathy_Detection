import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
from openai import OpenAI
import os
import time

# ตั้งค่าเริ่มต้น
st.set_page_config(
    page_title="EYE Care Pro",
    layout="wide",
    page_icon="👁️",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        background-color: #F8F9FA;
    }
    .header-text {
        color: #2E86C1;
        font-size: 2.5em;
        text-align: center;
        padding: 20px;
        border-bottom: 3px solid #2E86C1;
        margin-bottom: 30px;
    }
    .diagnosis-card {
        background: white;
        border-radius: 15px;
        padding: 25px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-top: 20px;
    }
    .loading-animation {
        display: flex;
        justify-content: center;
        font-size: 2em;
        color: #2E86C1;
    }
</style>
""", unsafe_allow_html=True)

# โหลดโมเดล
MODEL_PATH = 'CEDT_Model.h5'
try:
    model = load_model(MODEL_PATH, compile=False, safe_mode=False)
    TARGET_SIZE = model.input_shape[1:3]  # รับขนาดภาพจากโมเดล
except Exception as e:
    st.error(f"⚠️ โหลดโมเดลไม่สำเร็จ: {str(e)}")
    st.stop()

# ตั้งค่า Typhoon LLM
TYPHOON_CONFIG = {
    "api_key": "sk-3wY19YJQdjyYVBnwdjZKlpa3X7KG58tACnkPuAaH5rT8k70u",
    "base_url": "https://opentyphoon.ai/api/v1",
    "model": "typhoon-v1.5x-70b-instruct"
}

client = OpenAI(**TYPHOON_CONFIG)

# ฟังก์ชันประมวลผลภาพ
def preprocess_image(image):
    try:
        if image.mode != 'RGB':
            image = image.convert('RGB')
        img = image.resize(TARGET_SIZE, Image.Resampling.LANCZOS)
        return np.expand_dims(np.array(img)/255.0, axis=0)
    except Exception as e:
        st.error(f"⚠️ ข้อผิดพลาดในการประมวลผลภาพ: {str(e)}")
        return None

# ฟังก์ชันสร้างคำแนะนำ
def generate_medical_advice(diagnosis_level):
    PROMPT_TEMPLATE = """
    คุณคือจักษุแพทย์ผู้เชี่ยวชาญ โปรดให้คำแนะนำผู้ป่วยโรคจอประสาทตาเบาหวานระดับ {level} 
    โดยแบ่งเป็นหัวข้อต่อไปนี้:
    1. คำอธิบายระดับความรุนแรง
    2. แนวทางการรักษา
    3. ข้อควรปฏิบัติตัว
    4. ข้อห้าม
    5. คำแนะนำพิเศษ
    โปรดใช้ภาษาที่เข้าใจง่าย ไม่ใช้ศัพท์เทคนิคเกินจำเป็น
    """
    
    try:
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": PROMPT_TEMPLATE.format(level=diagnosis_level)},
                {"role": "user", "content": "สร้างคำแนะนำทางการแพทย์ตามระดับที่ระบุ"}
            ],
            temperature=0.3,
            max_tokens=600,
            top_p=0.95
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"⚠️ ข้อผิดพลาดในการสร้างคำแนะนำ: {str(e)}")
        return None

# ส่วนติดต่อผู้ใช้
st.markdown('<div class="header-text">👁️ EYE Care Pro - ระบบวิเคราะห์จอประสาทตาเบาหวาน</div>', unsafe_allow_html=True)

# ส่วนเลือกวิธีการป้อนข้อมูล
input_method = st.radio(
    "เลือกวิธีการป้อนข้อมูล:",
    ["📤 อัปโหลดภาพ", "📷 ถ่ายภาพด้วยกล้อง", "🖼️ ใช้ภาพตัวอย่าง"],
    horizontal=True,
    index=0
)

# การจัดการภาพ
image_source = None

if input_method == "📤 อัปโหลดภาพ":
    uploaded_file = st.file_uploader(" ", type=["jpg", "jpeg", "png"], key="uploader")
    if uploaded_file:
        image_source = Image.open(uploaded_file)

elif input_method == "📷 ถ่ายภาพด้วยกล้อง":
    camera_image = st.camera_input("ถ่ายภาพจอประสาทตา")
    if camera_image:
        image_source = Image.open(camera_image)

elif input_method == "🖼️ ใช้ภาพตัวอย่าง":
    sample_folder = "sample_images"
    if os.path.exists(sample_folder):
        samples = [f for f in os.listdir(sample_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if samples:
            selected = st.selectbox("เลือกภาพตัวอย่าง", samples)
            if selected:
                image_source = Image.open(os.path.join(sample_folder, selected))
        else:
            st.warning("⚠️ ไม่พบภาพตัวอย่างในโฟลเดอร์")

# ส่วนประมวลผลหลัก
if image_source:
    with st.container():
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # แสดงภาพต้นทาง
            st.image(
                image_source,
                caption="ภาพที่เลือก",
                use_container_width=True,
                output_format="JPEG"
            )
            
            # ปุ่มวิเคราะห์
            if st.button("🔍 เริ่มวิเคราะห์", type="primary", use_container_width=True):
                with st.spinner("กำลังประมวลผล..."):
                    try:
                        # โหลดและประมวลผลภาพ
                        processed_img = preprocess_image(image_source)
                        
                        if processed_img is not None:
                            # ทำนายผล
                            prediction = model.predict(processed_img)
                            diagnosis_level = np.argmax(prediction[0])
                            
                            # สร้างคำแนะนำ
                            medical_advice = generate_medical_advice(diagnosis_level)
                            
                            # บันทึกผลลัพธ์
                            st.session_state.last_diagnosis = {
                                "image": image_source,
                                "level": diagnosis_level,
                                "advice": medical_advice
                            }
                            st.rerun()
                            
                    except Exception as e:
                        st.error(f"⚠️ เกิดข้อผิดพลาด: {str(e)}")

        with col2:
            # แสดงผลลัพธ์
            if 'last_diagnosis' in st.session_state:
                diagnosis = st.session_state.last_diagnosis
                formatted_advice = diagnosis['advice'].replace('\n', '<br>')
                
                st.markdown(f"""
                <div class="diagnosis-card">
                    <h3>ระดับความรุนแรง: {diagnosis['level']}</h3>
                    <div style="margin-top:20px">
                        {formatted_advice}
                    </div>
                </div>
                """, unsafe_allow_html=True)

# ส่วนตั้งค่าใน sidebar
with st.sidebar:
    st.header("⚙️ การตั้งค่า")
    
    with st.expander("🧠 ตั้งค่า AI"):
        st.slider("ความสร้างสรรค์", 0.0, 1.0, 0.3, key="temperature")
        st.slider("ความเฉพาะเจาะจง", 0.0, 1.0, 0.95, key="top_p")
    
    with st.expander("📚 ประวัติการวิเคราะห์"):
        if 'diagnosis_history' in st.session_state:
            for idx, record in enumerate(st.session_state.diagnosis_history):
                with st.container():
                    st.image(record["image"], use_container_width=True)
                    st.markdown(f"**ระดับ {record['level']}**")
                    st.write(record["advice"])
                    st.divider()

# ข้อมูลระบบ
st.sidebar.markdown("""
<div style="margin-top:50px; color:#666">
    <small>
    ℹ️ ระบบนี้ใช้สำหรับการสนับสนุนการวินิจฉัยเบื้องต้นเท่านั้น<br>
    Version: 1.2.0 | Last updated: 2025-03-10
    </small>
</div>
""", unsafe_allow_html=True)