import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
from openai import OpenAI
import os
import time

# ตั้งค่าหน้าเว็บ
st.set_page_config(
    page_title="EYE Care Pro",
    layout="wide",
    page_icon="👁️",
    initial_sidebar_state="expanded"
)

# สไตล์ CSS สำหรับ UI
st.markdown("""
<style>
    .main {
        background-color: #F8F9FA;
        padding: 20px;
    }
    .header {
        color: #2E86C1;
        font-size: 2.5em;
        text-align: center;
        padding: 20px;
        border-bottom: 3px solid #2E86C1;
        margin-bottom: 30px;
    }
    .diagnosis-box {
        background: white;
        border-radius: 15px;
        padding: 25px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-top: 20px;
    }
    .image-preview {
        border: 2px solid #AED6F1;
        border-radius: 10px;
        padding: 10px;
        background: white;
    }
    .stButton>button {
        background: #3498DB !important;
        color: white !important;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background: #2980B9 !important;
        transform: scale(1.05);
    }
</style>
""", unsafe_allow_html=True)

# โหลดโมเดล AI
try:
    model = load_model('CEDT_Model.h5', compile=False, safe_mode=False)
    TARGET_SIZE = model.input_shape[1:3]  # รับขนาดภาพจากโมเดล
except Exception as e:
    st.error(f"⚠️ ไม่สามารถโหลดโมเดลได้: {str(e)}")
    st.stop()

# ตั้งค่า Deepseek API
DEEPSEEK_CONFIG = {
    "api_key": os.getenv("DEEPSEEK_API_KEY"),  # ตั้งค่า environment variable
    "base_url": "https://api.deepseek.com/v1"
}

client = OpenAI(**DEEPSEEK_CONFIG)

def preprocess_image(image):
    """ปรับขนาดและเตรียมภาพสำหรับโมเดล"""
    try:
        if image.mode != 'RGB':
            image = image.convert('RGB')
        img = image.resize(TARGET_SIZE, Image.Resampling.LANCZOS)
        return np.expand_dims(np.array(img)/255.0, axis=0)
    except Exception as e:
        st.error(f"⚠️ เกิดข้อผิดพลาดในการประมวลผลภาพ: {str(e)}")
        return None

def generate_advice(level):
    """สร้างคำแนะนำทางการแพทย์ด้วย Deepseek API"""
    prompt_template = f"""
    คุณคือจักษุแพทย์ผู้เชี่ยวชาญ โปรดให้คำแนะนำสำหรับผู้ป่วยโรคจอประสาทตาเบาหวานระดับ {level}
    โดยแบ่งออกเป็นส่วนดังนี้:
    1. อาการที่พบ
    2. แนวทางการรักษา
    3. ข้อควรปฏิบัติ
    4. อาหารแนะนำ
    5. ข้อห้าม
    6. คำแนะนำเพิ่มเติม
    โปรดใช้ภาษาที่เข้าใจง่าย พร้อมตัวอย่างการปฏิบัติตัว
    """
    
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "คุณเป็นแพทย์ผู้เชี่ยวชาญด้านจักษุวิทยา"},
                {"role": "user", "content": prompt_template}
            ],
            temperature=0.3,
            max_tokens=600,
            top_p=0.95
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"⚠️ เกิดข้อผิดพลาดในการสร้างคำแนะนำ: {str(e)}")
        return None

# ส่วนแสดงผล UI
st.markdown('<div class="header">👁️ EYE Care Pro - ระบบวิเคราะห์จอประสาทตาเบาหวาน</div>', unsafe_allow_html=True)

# เลือกแหล่งภาพ
input_method = st.radio(
    "เลือกวิธีการป้อนข้อมูล:",
    ["📤 อัปโหลดภาพ", "📷 ถ่ายภาพ", "🖼️ ภาพตัวอย่าง"],
    horizontal=True
)

image_source = None

# จัดการแหล่งภาพ
if input_method == "📤 อัปโหลดภาพ":
    uploaded_file = st.file_uploader(" ", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image_source = Image.open(uploaded_file)

elif input_method == "📷 ถ่ายภาพ":
    camera_img = st.camera_input("ถ่ายภาพตา")
    if camera_img:
        image_source = Image.open(camera_img)

elif input_method == "🖼️ ภาพตัวอย่าง":
    sample_folder = "sample_images"
    if os.path.exists(sample_folder):
        samples = [f for f in os.listdir(sample_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if samples:
            selected = st.selectbox("เลือกภาพตัวอย่าง", samples)
            if selected:
                image_source = Image.open(os.path.join(sample_folder, selected))
        else:
            st.warning("⚠️ ไม่พบภาพตัวอย่างในโฟลเดอร์")

# ประมวลผลภาพ
if image_source:
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown('<div class="image-preview">', unsafe_allow_html=True)
        st.image(image_source, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        if st.button("🔍 เริ่มวิเคราะห์", type="primary"):
            with st.spinner("กำลังวิเคราะห์..."):
                try:
                    processed_img = preprocess_image(image_source)
                    if processed_img is not None:
                        prediction = model.predict(processed_img)
                        diagnosis_level = np.argmax(prediction[0])
                        
                        advice = generate_advice(diagnosis_level)
                        
                        if advice:
                            st.session_state.diagnosis = {
                                "image": image_source,
                                "level": diagnosis_level,
                                "advice": advice
                            }
                            st.rerun()
                            
                except Exception as e:
                    st.error(f"⚠️ เกิดข้อผิดพลาด: {str(e)}")

    with col2:
        if 'diagnosis' in st.session_state:
            diagnosis = st.session_state.diagnosis
            formatted_advice = diagnosis['advice'].replace('\n', '<br>')
            
            st.markdown(f"""
            <div class="diagnosis-box">
                <h3>ผลการวินิจฉัย: ระดับ {diagnosis['level']}</h3>
                <div style="margin-top:20px; line-height:1.6">
                    {formatted_advice}
                </div>
            </div>
            """, unsafe_allow_html=True)

# ส่วนเสริม
with st.sidebar:
    st.header("⚙️ การตั้งค่า")
    
    with st.expander("⚡ ตั้งค่าการทำงาน"):
        st.slider("ความละเอียดการวิเคราะห์", 0.1, 1.0, 0.8, key="model_precision")
        st.slider("ความสร้างสรรค์คำแนะนำ", 0.0, 1.0, 0.5, key="creativity")
    
    with st.expander("📜 ประวัติการตรวจ"):
        if 'diagnosis' in st.session_state:
            st.image(st.session_state.diagnosis['image'], use_container_width=True)
            st.write(f"ระดับ: {st.session_state.diagnosis['level']}")
            st.write(st.session_state.diagnosis['advice'])

st.sidebar.markdown("""
<div style="color:#666; margin-top:50px">
    <small>
    ℹ️ ระบบนี้ใช้เพื่อการสนับสนุนการวินิจฉัยเบื้องต้นเท่านั้น<br>
    พัฒนาโดยทีม EYE Care - © 2025
    </small>
</div>
""", unsafe_allow_html=True)