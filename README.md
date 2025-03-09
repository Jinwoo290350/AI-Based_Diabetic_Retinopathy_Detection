# AI-Based_Diabetic_Retinopathy_Detection

โครงการนี้มีวัตถุประสงค์เพื่อพัฒนาโมเดลสำหรับตรวจจับโรคเบาหวานในรูม่านตาโดยใช้เทคนิค Deep Learning ที่สามารถช่วยในการวินิจฉัยและคัดกรองโรคเบาหวานในระยะเริ่มแรก

---

## รายละเอียดโครงการ

- **ชื่อโครงการ:** AI-Based_Diabetic_Retinopathy_Detection
- **จุดประสงค์:** พัฒนาโมเดลตรวจจับโรคเบาหวานในรูม่านตาเพื่อสนับสนุนนวัตกรรมทางการแพทย์
- **เครื่องมือและเทคโนโลยี:** Python, Jupyter Notebook, Keras/TensorFlow

---

## แหล่งข้อมูล (Datasets)

- **Diabetic Retinopathy Resized Dataset:**  
  โดนได้มาจาก [Kaggle Dataset](https://www.kaggle.com/datasets/tanlikesmath/diabetic-retinopathy-resized)  
  (Dataset นี้มีการปรับขนาดภาพเพื่อให้เหมาะสมกับการประมวลผลและเทรนโมเดล)

---
## explian DATA
- **Simple picture to explain Diabetic Retinopathy**
 How do we know that a patient have diabetic retinopahy? There are at least 5 things to spot on. Image credit https://www.eyeops.com/credit : 
 https://www.eyeops.com/

![image](https://github.com/user-attachments/assets/18afc44e-c4e8-4815-bbfa-e301a1214b25)

 From quick investigations of the data (see various pictures below), I found that Hemorrphages, Hard Exudates and Cotton Wool spots are quite easily 
 observed. However, I still could not find examples of Aneurysm or Abnormal Growth of Blood Vessels from our data yet. Perhaps the latter two cases are 
 important if we want to catch up human benchmnark using our model.
---

## โมเดล (Model)

- **Densenet Keras:**  
  ใช้สถาปัตยกรรม DenseNet ที่นำมาปรับใช้ผ่าน [Densenet Keras](https://www.kaggle.com/datasets/xhlulu/densenet-keras)  
  (โมเดลนี้ช่วยให้การวิเคราะห์ภาพมีความแม่นยำและประสิทธิภาพสูงในการตรวจจับสัญญาณของโรค)

---

## เครดิตข้อมูลการแข่งขัน (Data Competition Credit)

- โครงการนี้ได้รับแรงบันดาลใจและข้อมูลจากการแข่งขัน [APTOS 2019 Blindness Detection](https://www.kaggle.com/competitions/aptos2019-blindness-detection)  
  (ขอบคุณการแข่งขันและชุมชน Kaggle ที่ให้ข้อมูลและแนวคิดสำหรับการพัฒนาโมเดลนี้)

---

## โค้ดและการทำงาน

- **ไฟล์หลัก:** `cedt-hackathon-of-newbie.ipynb`  
  ไฟล์ Notebook นี้ประกอบด้วยขั้นตอนต่าง ๆ ดังนี้:
  - **Data Preprocessing:** การเตรียมข้อมูลและปรับขนาดภาพ
  - **Model Training:** การเทรนโมเดลโดยใช้สถาปัตยกรรม DenseNet
  - **Evaluation:** การประเมินผลโมเดลและปรับแต่งพารามิเตอร์

---

## วิธีการใช้งาน

1. **Clone Repository:**
   ```bash
   git clone https://github.com/Jinwoo290350/AI-Based_Diabetic_Retinopathy_Detection.git
   cd AI-Based_Diabetic_Retinopathy_Detection
