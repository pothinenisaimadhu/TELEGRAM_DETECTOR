## 🚀 Impactful Projects

### 1. Deepfake Detection using XceptionNet
**GitHub / Reference URL:** (https://github.com/pothinenisaimadhu/TELEGRAM_DETECTOR)

**Description:**  
Built a deep learning pipeline to detect **real vs. fake images** from the **Celeb-DF v2 dataset** using **PyTorch** and **XceptionNet**. The system classifies facial images as either genuine or AI-synthesized (deepfake), with training/validation workflows, model saving, and inference support.  

**Role & Contributions:**  
- Designed a **Custom PyTorch Dataset class** for preprocessing and labeling.  
- Implemented **training/validation pipeline** with augmentation and DataLoaders.  
- Fine-tuned **XceptionNet** (via `timm`) for binary classification with BCE loss.  
- Built a **full training loop** with accuracy/loss tracking and visualization.  
- Developed an **inference pipeline** to classify unseen images with confidence scores.  

**Challenges:**  
- Efficient handling of large dataset and memory optimization.  
- Tackled class imbalance and overfitting with augmentation/regularization.  
- Ensured preprocessing consistency across training and inference stages.  

**Impact:**  
- Achieved **~90% validation accuracy** on Celeb-DF dataset.  
- Delivered a **reusable pipeline** for training/testing on other deepfake datasets.  
- Enabled **real-time inference** of real vs. fake images with confidence scoring.  
- Supported **research in media authenticity**, mitigating misinformation risks.  

---

### 2. Road Damage Detection using YOLOv8
**GitHub / Reference URL:** [Add your repo link here]

**Description:**  
Implemented an **object detection system** using **YOLOv8** to identify and classify road surface damages (D00, D10, D20, D40) from images. Used the **RDD2020 dataset** for training, with annotations converted from XML to YOLO TXT format.  

**Role & Contributions:**  
- Preprocessed dataset and converted annotations to **YOLO format**.  
- Configured **YOLOv8 training pipeline** with custom YAML definitions.  
- Implemented **TensorBoard logging** for monitoring training/validation metrics.  
- Fine-tuned YOLOv8 for **multi-class damage detection**.  
- Built a **real-time inference script** for detecting damages from road images.  

**Challenges:**  
- Handling **large-scale dataset** with varied road conditions and lighting.  
- Balancing precision and recall across **four different damage classes**.  
- Optimizing GPU memory usage during training.  

**Impact:**  
- Achieved **high mAP scores (~85%)** across all damage classes.  
- Enabled **automated road damage detection**, reducing manual inspection effort.  
- Delivered a **deployable YOLOv8 model** for real-world road safety applications.  
- Provided insights for **smart city infrastructure** and maintenance planning.  
