# AI-Based Diabetic Retinopathy Detection

This project aims to develop a deep learning-based model for detecting diabetic retinopathy. The model is designed to assist in the early diagnosis and screening of diabetic retinopathy in patients.

---

## Project Details

- **Project Name:** AI-Based Diabetic Retinopathy Detection
- **Objective:** Develop a model to detect diabetic retinopathy to support medical innovation.
- **Tools and Technologies:** Python, Jupyter Notebook, Keras/TensorFlow

---

## Datasets

- **Diabetic Retinopathy Resized Dataset:**  
  Sourced from the [Kaggle Dataset](https://www.kaggle.com/datasets/tanlikesmath/diabetic-retinopathy-resized)  
  *(This dataset contains resized images to facilitate processing and model training)*

---

## Explain Data

- **Simple Illustration for Diabetic Retinopathy:**  
  How do we know if a patient has diabetic retinopathy? There are at least five key indicators to spot.  
  Image credit: [EyeOps](https://www.eyeops.com/)  
  ![image](https://github.com/user-attachments/assets/18afc44e-c4e8-4815-bbfa-e301a1214b25)

  From a preliminary investigation of the data (see various images below), I found that hemorrhages, hard exudates, and cotton wool spots are quite easily observed. However, I have not yet found examples of aneurysms or abnormal growth of blood vessels in our dataset. Perhaps these latter two cases are important if we wish to achieve human benchmark performance with our model.

---

## Model

- **Densenet Keras:**  
  We utilize the DenseNet architecture adapted from [Densenet Keras](https://www.kaggle.com/datasets/xhlulu/densenet-keras).  
  *(This model offers high accuracy and efficiency in detecting signs of diabetic retinopathy)*

---

## Data Competition Credit

- This project is inspired by and utilizes data from the [APTOS 2019 Blindness Detection competition](https://www.kaggle.com/competitions/aptos2019-blindness-detection).  
  *(We thank the competition organizers and the Kaggle community for providing the data and insights that helped shape this model)*

---

## Code and Workflow

- **Main File:** `cedt-hackathon-of-newbie.ipynb`  
  This Jupyter Notebook includes the following steps:
  - **Data Preprocessing:** Preparing and resizing images.
  - **Model Training:** Training the model using the DenseNet architecture.
  - **Evaluation:** Assessing model performance and fine-tuning parameters.

---

## How to Use

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/Jinwoo290350/AI-Based_Diabetic_Retinopathy_Detection.git
   cd AI-Based_Diabetic_Retinopathy_Detection
