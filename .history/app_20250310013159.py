import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
from openai import OpenAI
import os

# ตั้งค่าเริ่มต้น
st.set_page_config(page_title="EYE", layout="wide")

# โหลดโมเดล classification
try:
    model = load_model('CEDT_Model.h5', compile=False, safe_mode=False)
    # ตรวจสอบขนาด Input ของโมเดล
    model_input_shape = model.input_shape[1:3]
except Exception as e:
    st.error(f"โหลดโมเดลไม่สำเร็จ: {str(e)}")
    st.stop()

# ตั้งค่า OpenAI Client
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url="https://api.opentyphoon.ai/v1",
)

# ฟังก์ชันประมวลผลภาพ
def preprocess_image(image):
    target_size = model_input_shape or (320, 320)  # ใช้ขนาดจากโมเดลหรือค่า default
    img = image.resize(target_size)
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
st.title("Welcome to our webapp!!")
st.caption("AI Application for supporting post-surgery patient recovery")

# ตั้งค่าใน Sidebar
with st.sidebar:
    st.header("Config")
    
    # เลือกโมเดล
    model_type = st.selectbox(
        "Classify Model",
        options=["Default Model", "Alternative Model"]
    )
    
    # ปรับพารามิเตอร์
    temperature = st.slider("Temperature", 0.0, 1.0, 0.6)
    top_p = st.slider("Top P", 0.0, 1.0, 0.95)
    max_tokens = st.slider("Max Token", 50, 512, 512)

# ส่วนอัปโหลดภาพและตัวอย่าง
st.header("Take a picture")

# เลือกตัวอย่างภาพ
sample_images = []
sample_folder = "./sample_images"
if os.path.exists(sample_folder):
    sample_images = [f for f in os.listdir(sample_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

col1, col2 = st.columns(2)
with col1:
    uploaded_file = st.file_uploader(
        "Upload your photo",
        type=["jpg", "jpeg", "png"],
        help="Limit 200MB per file - JPG, PNG, JPEG"
    )

with col2:
    if sample_images:
        selected_sample = st.selectbox("Or select a sample image", sample_images)
    else:
        st.warning("No sample images found in 'sample_images' folder")

# เก็บประวัติการวินิจฉัย
if 'diagnosis_history' not in st.session_state:
    st.session_state.diagnosis_history = []

# เมื่อมีภาพอัปโหลดหรือเลือกตัวอย่าง
image_source = None
if uploaded_file:
    image_source = Image.open(uploaded_file)
elif sample_images and selected_sample:
    image_path = os.path.join(sample_folder, selected_sample)
    image_source = Image.open(image_path)

# หน้าจอแสดงผลหลัก
if image_source:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Input Image")
        st.image(image_source, use_column_width=True)
        
        if st.button("Process Image"):
            with st.spinner("กำลังวิเคราะห์ภาพ..."):
                # Classification
                processed_img = preprocess_image(image_source)
                try:
                    prediction = model.predict(processed_img)
                    user_eye = np.argmax(prediction[0])
                    
                    # สร้างคำแนะนำ
                    advice = generate_advice(user_eye, temperature, top_p)
                    
                    # บันทึกประวัติ
                    st.session_state.diagnosis_history.append({
                        "image": image_source,
                        "level": user_eye,
                        "advice": advice
                    })
                except Exception as e:
                    st.error(f"เกิดข้อผิดพลาดในการประมวลผล: {str(e)}")

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

# กล้องเว็บแคม
st.header("Webcam Capture")
camera_image = st.camera_input("Take a photo with your webcam")

if camera_image:
    image_source = Image.open(camera_image)
    st.image(image_source, caption="Captured Image", use_column_width=True)

# แสดงคำเตือนเกี่ยวกับขนาดภาพ
if model_input_shape:
    st.warning(f"โมเดลนี้ต้องการภาพขนาด {model_input_shape[0]}x{model_input_shape[1]} พิกเซล")
else:
    st.warning("ไม่พบข้อมูลขนาดภาพที่โมเดลต้องการ")