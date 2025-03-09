import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
from openai import OpenAI
import os

# ตั้งค่าเริ่มต้น
st.set_page_config(page_title="SurgiCare", layout="wide")

# โหลดโมเดล classification
try:
    model = load_model('model.h5')  # เปลี่ยนชื่อไฟล์ตามจริง
except Exception as e:
    st.error(f"โหลดโมเดลไม่สำเร็จ: {str(e)}")
    st.stop()

# ตั้งค่า OpenAI Client
client = OpenAI(
    api_key=os.getenv("TYPHOON_API_KEY"),  # ควรเก็บ API Key ใน environment variable
    base_url="https://api.opentyphoon.ai/v1",
)

# ฟังก์ชันประมวลผลภาพ
def preprocess_image(image):
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

# ฟังก์ชันสร้างคำแนะนำจาก OpenAI
def generate_advice(user_eye, temperature, top_p):
    stream = client.chat.completions.create(
        model="typhoon-v1.5x-70b-instruct",
        messages=[
            {
                "role": "system",
                "content": """คุณคือแพทย์ผู้เชี่ยวชาญ ให้คำแนะนำการดูแลเบื้องต้นสำหรับโรคตาเบาหวานตามระดับความรุนแรง 0-4:
- 0: ไม่มีอาการ
- 1: เริ่มมีอาการ
- 2: ปานกลาง
- 3: รุนแรง
- 4: รุนแรงมาก
ให้คำแนะนำเป็นข้อๆ พร้อมระบุระดับอาการ"""
            },
            {"role": "user", "content": str(user_eye)}
        ],
        max_tokens=512,
        temperature=temperature,
        top_p=top_p,
        stream=True,
    )
    
    response = []
    for chunk in stream:
        if chunk.choices[0].delta.content:
            response.append(chunk.choices[0].delta.content)
    return ''.join(response)

# ส่วนติดต่อผู้ใช้
st.title("Welcome to SurgiCare!!")
st.caption("AI Application for supporting post-surgery patient recovery")

# ตั้งค่าใน Sidebar
with st.sidebar:
    st.header("Config")
    
    # เลือกโมเดล
    model_type = st.selectbox(
        "Classify Model",
        ["SurgiCare-V1-large-turbo", "SurgiCare-V2-enhanced"]
    )
    
    # ปรับพารามิเตอร์
    temperature = st.slider("Temperature", 0.0, 1.0, 0.6)
    top_p = st.slider("Top P", 0.0, 1.0, 0.95)
    max_tokens = st.slider("Max Token", 50, 512, 512)

# ส่วนอัปโหลดภาพ
st.header("Take a picture")
uploaded_file = st.file_uploader(
    "Upload or select a sample photo",
    type=["jpg", "jpeg", "png"],
    help="Limit 200MB per file - JPG, PNG, JPEG"
)

# เก็บประวัติการวินิจฉัย
if 'diagnosis_history' not in st.session_state:
    st.session_state.diagnosis_history = []

# เมื่อมีภาพอัปโหลด
if uploaded_file:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Uploaded Image")
        image = Image.open(uploaded_file)
        st.image(image, use_column_width=True)
        
        if st.button("Process Image"):
            with st.spinner("กำลังวิเคราะห์ภาพ..."):
                # Classification
                processed_img = preprocess_image(image)
                prediction = model.predict(processed_img)
                user_eye = np.argmax(prediction[0])
                
                # สร้างคำแนะนำ
                advice = generate_advice(user_eye, temperature, top_p)
                
                # บันทึกประวัติ
                st.session_state.diagnosis_history.append({
                    "image": image,
                    "level": user_eye,
                    "advice": advice
                })

    with col2:
        if st.session_state.diagnosis_history:
            latest = st.session_state.diagnosis_history[-1]
            st.subheader(f"Diagnosis Result (Level {latest['level']})")
            st.write(latest["advice"])

# ส่วนแสดงประวัติ
st.header("Full Diagnose History")
for idx, record in enumerate(st.session_state.diagnosis_history):
    with st.expander(f"Diagnosis #{idx+1} - Level {record['level']}"):
        col1, col2 = st.columns([1, 3])
        with col1:
            st.image(record["image"], width=150)
        with col2:
            st.write(record["advice"])