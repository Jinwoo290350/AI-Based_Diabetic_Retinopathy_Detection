import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
import requests
import os
import time
import base64
from io import BytesIO

# Set up the page
st.set_page_config(
    page_title="EYE Care Pro",
    layout="wide",
    page_icon="👁️",
    initial_sidebar_state="expanded"
)

# Updated CSS for the UI
st.markdown("""
<style>
    /* ระบบสีใหม่ */
    :root {
        --primary: #2565AE;
        --secondary: #3AB0FF;
        --accent: #FFB562;
        --text: #2D4356;
        --background: #F5F5F5;
    }

    .main {
        background: var(--background);
        padding: 2rem;
    }

    .header {
        color: var(--primary);
        font-family: 'Kanit', sans-serif;
        font-size: 2.75rem;
        text-align: center;
        padding: 1.5rem;
        border-bottom: 4px solid var(--secondary);
        margin-bottom: 2rem;
        letter-spacing: -0.5px;
    }

    /* การ์ดภาพถ่าย */
    .image-preview {
        border: 3px solid var(--secondary);
        border-radius: 20px;
        padding: 1rem;
        background: white;
        box-shadow: 0 8px 16px rgba(0,0,0,0.1);
        transition: transform 0.3s ease;
    }

    .image-preview:hover {
        transform: translateY(-5px);
    }

    /* กล่องผลลัพธ์ */
    .diagnosis-box {
        background: linear-gradient(145deg, #ffffff, #f8f9fa);
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 12px 24px rgba(0,0,0,0.08);
        margin-top: 2rem;
        border: 2px solid var(--secondary);
    }

    .diagnosis-box h3 {
        color: var(--primary) !important;
        font-size: 1.8rem;
        margin-bottom: 1.5rem;
        padding-bottom: 1rem;
        border-bottom: 2px solid var(--accent);
    }

    /* ปุ่มแบบอินเทอร์แอคทีฟ */
    .stButton>button {
        background: var(--primary) !important;
        color: white !important;
        border-radius: 12px !important;
        padding: 12px 24px !important;
        font-size: 1.1rem !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        border: none !important;
    }

    .stButton>button:hover {
        background: var(--secondary) !important;
        transform: scale(1.05) rotate(-2deg);
        box-shadow: 0 8px 16px rgba(58,176,255,0.3) !important;
    }

    /* เมนูเลือกภาพ */
    [data-testid="stHorizontalBlock"] {
        gap: 1rem !important;
    }

    /* ไซด์บาร์โมเดิร์น */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, var(--primary), #1A4D8F);
        padding: 2rem 1.5rem !important;
    }

    [data-testid="stSidebar"] h1 {
        color: white !important;
        font-family: 'Kanit', sans-serif;
        border-bottom: 2px solid var(--accent);
        padding-bottom: 1rem;
    }

    /* เอฟเฟคโหลดข้อมูล */
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }

    .spinner {
        animation: pulse 1.5s infinite;
        font-size: 1.5rem;
        color: var(--primary);
    }

    /* การแจ้งเตือนข้อผิดพลาด */
    .error-box {
        background: #FFE3E3;
        color: #CC0000;
        padding: 1.5rem;
        border-radius: 12px;
        border: 2px solid #FFAAAA;
        margin: 1.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Display header
st.markdown('<div class="header">👁️ EYE Care Pro - ระบบวิเคราะห์จอประสาทตาเบาหวาน</div>', unsafe_allow_html=True)

# Helper function to convert images to base64 for HTML <img> display
def get_image_as_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{img_str}"

# Load the AI model
try:
    model = load_model('CEDT_Model.h5', compile=False, safe_mode=False)
    TARGET_SIZE = model.input_shape[1:3]
except Exception as e:
    st.error(f"⚠️ ไม่สามารถโหลดโมเดลได้: {str(e)}")
    st.stop()

# Set up Typhoon API parameters
TYPHOON_API_KEY = "sk-3wY19YJQdjyYVBnwdjZKlpa3X7KG58tACnkPuAaH5rT8k70u"
TYPHOON_API_URL = "https://api.opentyphoon.ai/v1/chat/completions"

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {TYPHOON_API_KEY}"
}

def preprocess_image(image):
    """ปรับขนาดและเตรียมภาพสำหรับโมเดล"""
    try:
        if image.mode != 'RGB':
            image = image.convert('RGB')
        img = image.resize(TARGET_SIZE, Image.Resampling.LANCZOS)
        return np.expand_dims(np.array(img) / 255.0, axis=0)
    except Exception as e:
        st.markdown(f'<div class="error-box">⚠️ เกิดข้อผิดพลาดในการประมวลผลภาพ: {str(e)}</div>', unsafe_allow_html=True)
        return None

def generate_medical_advice(level):
    """สร้างคำแนะนำทางการแพทย์ด้วย Typhoon API"""
    prompt = f"""
    คุณคือจักษุแพทย์ผู้เชี่ยวชาญ โปรดให้คำแนะนำสำหรับผู้ป่วยโรคจอประสาทตาเบาหวานระดับ {level}
    โดยแบ่งเป็นหัวข้อต่อไปนี้:
    1. อาการที่พบ (อธิบายด้วยภาษาง่ายๆ)
    2. แนวทางการรักษา (ระบุวิธีการรักษาหลัก)
    3. ข้อควรปฏิบัติตัว (รายละเอียดการดูแลตนเอง)
    4. อาหารแนะนำ (ตัวอย่างเมนูอาหาร)
    5. ข้อห้าม (สิ่งต้องหลีกเลี่ยง)
    6. คำแนะนำเพิ่มเติม (เคล็ดลับการดูแลสุขภาพตา)
    
    โปรดใช้ภาษาที่เข้าใจง่าย ไม่ใช้ศัพท์เทคนิคเกินจำเป็น
    """
    
    try:
        payload = {
            "model": "typhoon-v1.5x-70b-instruct",
            "messages": [
                {"role": "system", "content": "คุณเป็นผู้เชี่ยวชาญด้านจักษุวิทยา"},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.3,
            "max_tokens": 800,
            "top_p": 0.95
        }
        
        response = requests.post(
            TYPHOON_API_URL,
            headers=headers,
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content']
        else:
            error_msg = f"⚠️ ข้อผิดพลาด API (รหัส {response.status_code}): {response.text}"
            st.markdown(f'<div class="error-box">{error_msg}</div>', unsafe_allow_html=True)
            return None
            
    except Exception as e:
        error_msg = f"⚠️ เกิดข้อผิดพลาดในการเชื่อมต่อ: {str(e)}"
        st.markdown(f'<div class="error-box">{error_msg}</div>', unsafe_allow_html=True)
        return None

# Image input selection using radio buttons
input_method = st.radio(
    "เลือกวิธีการป้อนข้อมูล:",
    ["📤 อัปโหลดภาพ", "📷 ถ่ายภาพ", "🖼️ ภาพตัวอย่าง"],
    horizontal=True,
    label_visibility="collapsed"
)

image_source = None

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

# Process the image and display results
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
                        
                        advice = generate_medical_advice(diagnosis_level)
                        
                        if advice:
                            # Convert image to base64 for display in the sidebar history
                            image_base64 = get_image_as_base64(image_source)
                            st.session_state.diagnosis = {
                                "image": image_base64,
                                "level": diagnosis_level,
                                "advice": advice
                            }
                            st.experimental_rerun()
                            
                except Exception as e:
                    error_msg = f"⚠️ เกิดข้อผิดพลาดในการวิเคราะห์: {str(e)}"
                    st.markdown(f'<div class="error-box">{error_msg}</div>', unsafe_allow_html=True)

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

# Sidebar configuration and history
with st.sidebar:
    st.header("⚙️ การตั้งค่า")
    
    with st.expander("🎛️ ตั้งค่าการทำงาน", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.slider("ความละเอียด", 0.1, 1.0, 0.8, key="model_precision")
        with col2:
            st.slider("ความสร้างสรรค์", 0.0, 1.0, 0.5, key="creativity")
    
    with st.expander("📜 ประวัติการตรวจ"):
        if 'diagnosis' in st.session_state:
            st.markdown(f"""
            <div style="background: rgba(255,255,255,0.1); padding: 1rem; border-radius: 12px; margin: 0.5rem 0;">
                <small>{time.strftime('%d/%m/%Y %H:%M')}</small>
                <p style="color: white; margin: 0.5rem 0;">ระดับ: {st.session_state.diagnosis['level']}</p>
                <img src="{st.session_state.diagnosis['image']}" style="width: 100%; border-radius: 8px;">
            </div>
            """, unsafe_allow_html=True)

st.sidebar.markdown("""
<div style="color:#EEE; margin-top:50px">
    <small>
    ℹ️ ระบบนี้ใช้เพื่อการสนับสนุนการวินิจฉัยเบื้องต้นเท่านั้น<br>
    พัฒนาโดยทีม EYE Care - © 2025<br>
    ใช้ Typhoon LLM สำหรับการสร้างคำแนะนำ
    </small>
</div>
""", unsafe_allow_html=True)
