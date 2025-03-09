import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
from openai import OpenAI
import os
import time

# ตั้งค่าเริ่มต้น
st.set_page_config(page_title="EYE Care", layout="wide", page_icon="👁️")

# Custom CSS
st.markdown("""
<style>
    .header-style { 
        font-size:35px !important; 
        color: #2980B9 !important;
        padding-bottom:20px;
    }
    .diagnosis-box {
        background-color: #E8F8F5;
        padding: 20px;
        border-radius: 10px;
        margin-top: 20px;
    }
    .image-preview {
        border: 2px solid #AED6F1;
        border-radius: 10px;
        padding: 10px;
    }
</style>
""", unsafe_allow_html=True)

# โหลดโมเดล
try:
    model = load_model('CEDT_Model.h5', compile=False, safe_mode=False)
    model_input_shape = model.input_shape[1:3]
except Exception as e:
    st.error(f"เกิดข้อผิดพลาดในการโหลดโมเดล: {str(e)}")
    st.stop()

# ตั้งค่า OpenAI
client = OpenAI(
    api_key=os.getenv(""),
    base_url="https://api.opentyphoon.ai/v1",
)

# ฟังก์ชันประมวลผลภาพ
def preprocess_image(image):
    target_size = model_input_shape or (320, 320)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    img = image.resize(target_size, Image.Resampling.LANCZOS)
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

# ฟังก์ชันสร้างคำแนะนำ
def generate_advice(user_eye):
    try:
        stream = client.chat.completions.create(
            model="typhoon-v1.5x-70b-instruct",
            messages=[{
                "role": "system",
                "content": """คุณคือแพทย์ผู้เชี่ยวชาญ ให้คำแนะนำการดูแลเบื้องต้นสำหรับโรคตาเบาหวานตามระดับความรุนแรง 0-4"""
            }, {
                "role": "user", 
                "content": str(user_eye)
            }],
            max_tokens=512,
            temperature=st.session_state.temperature,
            top_p=st.session_state.top_p,
            stream=True,
        )
        
        response = []
        for chunk in stream:
            if chunk.choices[0].delta.content:
                response.append(chunk.choices[0].delta.content)
        return ''.join(response)
    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในการสร้างคำแนะนำ: {str(e)}")
        return None

# ส่วนติดต่อผู้ใช้
st.markdown('<p class="header-style">👁️ EYE Care - ระบบวิเคราะห์จอประสาทตาเบาหวาน</p>', unsafe_allow_html=True)

# ส่วนเลือกข้อมูล
input_method = st.radio("เลือกวิธีการป้อนข้อมูล:", 
                       ["อัปโหลดภาพ", "ใช้ภาพตัวอย่าง"], 
                       horizontal=True,
                       key="input_method")

# การจัดการภาพ
image_source = None
if input_method == "อัปโหลดภาพ":
    uploaded_file = st.file_uploader(" ", type=["jpg", "jpeg", "png"], key="uploader")
    if uploaded_file:
        image_source = Image.open(uploaded_file)

elif input_method == "ใช้ภาพตัวอย่าง":
    sample_folder = "sample_images"
    if os.path.exists(sample_folder):
        sample_images = [f for f in os.listdir(sample_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if sample_images:
            selected_sample = st.selectbox("เลือกภาพตัวอย่าง", sample_images)
            if selected_sample:
                image_path = os.path.join(sample_folder, selected_sample)
                image_source = Image.open(image_path)
        else:
            st.warning("ไม่พบภาพตัวอย่างในโฟลเดอร์")

# ส่วนประมวลผล
if image_source:
    with st.container():
        col1, col2 = st.columns([1, 2])
        
        # คอลัมน์ซ้าย - ภาพและปุ่ม
        with col1:
            st.markdown('<div class="image-preview">', unsafe_allow_html=True)
            st.image(image_source, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            if st.button("เริ่มวิเคราะห์", type="primary"):
                with st.spinner("กำลังประมวลผล..."):
                    try:
                        # ประมวลผลภาพ
                        processed_img = preprocess_image(image_source)
                        prediction = model.predict(processed_img)
                        user_eye = np.argmax(prediction[0])
                        
                        # สร้างคำแนะนำ
                        advice = generate_advice(user_eye)
                        
                        if advice:
                            # บันทึกผลลัพธ์
                            if 'diagnosis_history' not in st.session_state:
                                st.session_state.diagnosis_history = []
                                
                            st.session_state.diagnosis_history.append({
                                "image": image_source,
                                "level": user_eye,
                                "advice": advice
                            })
                            
                            # แสดงผลลัพธ์ทันที
                            st.rerun()
                            
                    except Exception as e:
                        st.error(f"เกิดข้อผิดพลาด: {str(e)}")

        # คอลัมน์ขวา - ผลลัพธ์
        with col2:
            if 'diagnosis_history' in st.session_state and st.session_state.diagnosis_history:
                latest = st.session_state.diagnosis_history[-1]
                st.markdown(f'<div class="diagnosis-box">'
                            f'<h3>ผลการวินิจฉัย (ระดับ {latest["level"]})</h3>'
                            f'{latest["advice"]}'
                            f'</div>', unsafe_allow_html=True)

# ส่วนตั้งค่าใน sidebar
with st.sidebar:
    st.header("การตั้งค่า")
    
    with st.expander("ตั้งค่า AI"):
        st.session_state.temperature = st.slider("Temperature", 0.0, 1.0, 0.6)
        st.session_state.top_p = st.slider("Top P", 0.0, 1.0, 0.95)
        st.session_state.max_tokens = st.slider("Max Tokens", 50, 512, 512)

    with st.expander("ประวัติการวิเคราะห์"):
        if 'diagnosis_history' in st.session_state and st.session_state.diagnosis_history:
            for idx, record in enumerate(st.session_state.diagnosis_history):
                st.subheader(f"การวิเคราะห์ #{idx+1}")
                st.image(record["image"], use_container_width=True)
                st.write(f"ระดับ: {record['level']}")
                st.write(record["advice"])
                st.divider()

# แสดงคำเตือนขนาดภาพ
if model_input_shape:
    st.info(f"⚠️ โมเดลนี้ต้องการภาพขนาด {model_input_shape[0]}x{model_input_shape[1]} พิกเซล")