
## 📌 Task 2 — Plant Disease Image Classifier (CNN with TensorFlow or PyTorch)

### 🎯 Goal
Train an image classifier that detects plant disease in **bean leaves**:
- Healthy  
- Bean Rust  
- Angular Leaf Spot  

### 🔧 Tools
- TensorFlow **or** PyTorch + torchvision  
- matplotlib  
- scikit-learn  

### 📂 Dataset
- [Beans Dataset – Hugging Face](https://huggingface.co/datasets/beans)  
- ~1,000 leaf images categorized into 3 classes.

### 🚀 Steps
1. Download dataset and organize folders:
2. Load dataset:
- `ImageDataGenerator` (TensorFlow)  
- OR `ImageFolder` (PyTorch)  
3. Resize images → **128x128**.  
4. Normalize pixel values → `[0, 1]`.  
5. Build a CNN:
- Conv → ReLU → MaxPool  
- Conv → ReLU → MaxPool  
- Dense → Softmax (3 classes)  
6. Train for **10–15 epochs**.  
7. Evaluate on test set:
- Accuracy  
- Confusion Matrix  
8. Plot **training loss & accuracy curves**.  
9. Save trained model → `.h5` (TensorFlow) or `.pt` (PyTorch).  
10. Write a script to load model and predict from an **image file**.  

### 📦 Deliverables
- `plant_classifier.py` → Training code  
- `plant_model.h5` / `plant_model.pt` → Saved model  
- `predict_image.py` → Script to test single image