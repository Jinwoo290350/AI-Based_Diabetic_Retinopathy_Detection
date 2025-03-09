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
</style>
""", unsafe_allow_html=True)

# โหลดโมเดล
MODEL_PATH = 'CEDT_Model.h5'
try:
    model = load_model(MODEL_PATH, compile=False, safe_mode=False)
    TARGET_SIZE = model.input_shape[1:3]
except Exception as e:
    st.error(f"⚠️ โหลดโมเดลไม่สำเร็จ: {str(e)}")
    st.stop()

# ตั้งค่า Typhoon LLM
TYPHOON_CONFIG = {
    "api_key": "Tsk-3wY19YJQdjyYVBnwdjZKlpa3X7KG58tACnkPuAaH5rT8k70u",
    "base_url": "https://opentyphoon.ai/api/v1"
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
            model="typhoon-v1.5x-70b-instruct",  # ระบุ model ในนี้
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

# ส่วนติดต่อผู้ใช้และส่วนอื่นๆ เหมือนเดิม...