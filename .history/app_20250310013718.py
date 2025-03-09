import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
from openai import OpenAI
import os

# ตั้งค่าเริ่มต้น
st.set_page_config(page_title="EYE Care", layout="wide", page_icon="👁️")

# Custom CSS สำหรับปรับแต่ง UI
st.markdown("""
<style>
    .header-style { font-size:35px !important; color:#2E86C1 !important; padding-bottom:20px; }
    .subheader-style { font-size:25px !important; color:#148F77 !important; }
    .success-box { background-color:#E8F8F5; padding:20px; border-radius:10px; margin:10px 0; }
    .warning-box { background-color:#FDEBD0; padding:20px; border-radius:10px; margin:10px 0; }
    .image-container { border:2px solid #AED6F1; border-radius:10px; padding:10px; }
</style>
""", unsafe_allow_html=True)

# โหลดโมเดล classification
try:
    model = load_model('CEDT_Model.h5', compile=False, safe_mode=False)
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
    target_size = model_input_shape or (320, 320)
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
st.markdown('<p class="header-style">👁️ EYE Care - Diabetic Retinopathy Detection</p>', unsafe_allow_html=True)

# ส่วนเลือก Input แบบแท็บ
input_method = st.radio("เลือกวิธีการป้อนข้อมูล:", 
                       ["อัปโหลดภาพ", "ใช้ภาพตัวอย่าง"], 
                       horizontal=True,
                       label_visibility="collapsed")

image_source = None

if input_method == "อัปโหลดภาพ":
    uploaded_file = st.file_uploader("ลากไฟล์ภาพมาที่นี่หรือคลิกเพื่อเลือกไฟล์", 
                                    type=["jpg", "jpeg", "png"],
                                    help="รองรับไฟล์ JPG, PNG, JPEG ขนาดไม่เกิน 200MB")
    if uploaded_file:
        image_source = Image.open(uploaded_file)

elif input_method == "ใช้ภาพตัวอย่าง":
    sample_folder = "sample_images"
    if os.path.exists(sample_folder):
        sample_images = [f for f in os.listdir(sample_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if sample_images:
            cols = st.columns(3)
            for idx, img_file in enumerate(sample_images):
                with cols[idx % 3]:
                    image_path = os.path.join(sample_folder, img_file)
                    img = Image.open(image_path)
                    if st.button(f"ตัวอย่าง {idx+1}"):
                        image_source = img
                        st.session_state.selected_sample = img_file
            if 'selected_sample' in st.session_state:
                st.markdown(f'<div class="success-box">✅ กำลังใช้ภาพตัวอย่าง: {st.session_state.selected_sample}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="warning-box">⚠️ ไม่พบภาพตัวอย่างในโฟลเดอร์ sample_images</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="warning-box">⚠️ โฟลเดอร์ sample_images ไม่พบในระบบ</div>', unsafe_allow_html=True)

# ส่วนแสดงภาพและประมวลผล
if image_source:
    with st.container():
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown('<p class="subheader-style">ภาพที่เลือก</p>', unsafe_allow_html=True)
            st.markdown('<div class="image-container">', unsafe_allow_html=True)
            st.image(image_source, use_column_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            if st.button("🏥 เริ่มวิเคราะห์ภาพ", type="primary", use_container_width=True):
                with st.spinner("🔍 กำลังวิเคราะห์ภาพ..."):
                    try:
                        processed_img = preprocess_image(image_source)
                        prediction = model.predict(processed_img)
                        user_eye = np.argmax(prediction[0])
                        
                        advice = generate_advice(user_eye, 
                                                st.session_state.get('temperature', 0.6), 
                                                st.session_state.get('top_p', 0.95))
                        
                        st.session_state.diagnosis_history.append({
                            "image": image_source,
                            "level": user_eye,
                            "advice": advice
                        })
                        st.success("✅ การวิเคราะห์เสร็จสมบูรณ์!")
                    except Exception as e:
                        st.error(f"❌ เกิดข้อผิดพลาด: {str(e)}")

        with col2:
            if st.session_state.get('diagnosis_history'):
                latest = st.session_state.diagnosis_history[-1]
                st.markdown(f'<p class="subheader-style">ผลการวินิจฉัย (ระดับ {latest["level"]})</p>', unsafe_allow_html=True)
                st.markdown(f'<div class="success-box">{latest["advice"]}</div>', unsafe_allow_html=True)

# ส่วนการตั้งค่าใน Sidebar
with st.sidebar:
    st.markdown('<p class="header-style">⚙️ การตั้งค่า</p>', unsafe_allow_html=True)
    
    # การตั้งค่าโมเดล
    with st.expander("🧠 ตั้งค่าโมเดล"):
        model_type = st.selectbox("ประเภทโมเดล", ["โมเดลมาตรฐาน", "โมเดลทางเลือก"])
        
    # การตั้งค่า AI
    with st.expander("🤖 ตั้งค่า AI"):
        st.session_state.temperature = st.slider("Temperature", 0.0, 1.0, 0.6)
        st.session_state.top_p = st.slider("Top P", 0.0, 1.0, 0.95)
        st.session_state.max_tokens = st.slider("Max Token", 50, 512, 512)
    
    # ประวัติการวินิจฉัย
    with st.expander("📜 ประวัติการวิเคราะห์"):
        if 'diagnosis_history' not in st.session_state:
            st.session_state.diagnosis_history = []
            
        for idx, record in enumerate(st.session_state.diagnosis_history):
            st.write(f"📌 การวิเคราะห์ #{idx+1} (ระดับ {record['level']})")
            st.image(record["image"], width=100)
            if st.button(f"แสดงรายละเอียด #{idx+1}"):
                st.write(record["advice"])

# แสดงคำเตือนเกี่ยวกับขนาดภาพ
if model_input_shape:
    st.markdown(f'<div class="warning-box">⚠️ โมเดลนี้ต้องการภาพขนาด {model_input_shape[0]}x{model_input_shape[1]} พิกเซล</div>', unsafe_allow_html=True)