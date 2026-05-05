# 🧠 Deepfake Detection System

An AI-powered web application that detects whether an image is **Real or Fake (Deepfake)** using Deep Learning.

---

## 🚀 Project Overview

This project uses **Transfer Learning with MobileNetV2** to classify images as real or fake.  
It includes a complete pipeline from training to deployment using a **Streamlit web interface**.

---

## 🎯 Features

✔️ Upload image and get prediction  
✔️ Detect Real vs Fake faces  
✔️ Confidence score display  
✔️ Clean and interactive UI (Streamlit)  
✔️ Uses pretrained deep learning model  

---

## 🧠 Model Details

- Base Model: **MobileNetV2 (Pretrained on ImageNet)**
- Technique: **Transfer Learning**
- Input Size: `96 x 96`
- Output: Binary Classification (Real / Fake)

---

## ⚙️ Technologies Used

- Python
- TensorFlow / Keras
- OpenCV
- NumPy
- Matplotlib
- Streamlit

---

## 📂 Project Structure
Deepfake-Detection/
│── train.py # Model training script
│── predict.py # Prediction logic
│── app.py # Streamlit frontend
│── requirements.txt # Dependencies
│── best_model.keras # Best trained model (if included)
│── README.md


---

## ▶️ How to Run the Project

### 1️⃣ Clone the repository
git clone (https://github.com/snehaagrawal34/Deepfake-Image-Detection-Model.git

cd deepfake-detection


### 2️⃣ Install dependencies

pip install -r requirements.txt


### 3️⃣ Run the app

streamlit run app.py


---

## 🧪 Model Training

To train the model manually:


python train.py


This will generate the trained model file.

---


## 🔧 Future Improvements

- Increase dataset size
- Add face detection before classification
- Train with GPU for better accuracy
- Fine-tune MobileNet layers

---

## 📌 Note

If `best_model.keras` is not included due to size limitations:

👉 Run `train.py` to generate the model manually.

---

## 👩‍💻 Authors

- Sneha Agrawal  
## 📚 References

- TensorFlow Documentation: https://www.tensorflow.org/
- OpenCV: https://opencv.org/
- Kaggle Datasets: https://www.kaggle.com/

---

## ⭐ Conclusion

This project demonstrates how deep learning can be applied to detect deepfake images and highlights the importance of AI in combating misinformation.
