# Machine Learning Final Project

## Battle of the Networks: Comparing ResNet50 and DenseNet121 for Pneumonia Detection from Chest X-Rays


## Tools and Frameworks

- **TensorFlow 2.x (Keras API)** – Model development and training  
- **Python** – Data processing and experimentation  
- **NumPy & Pandas** – Data handling  
- **Matplotlib & Seaborn** – Visualization  
- **OpenCV** – Image preprocessing  
- **Keras Tuner** – Hyperparameter optimization  

---

## 1. Dataset

For this project, we used the **Chest X-Ray (Pneumonia)** dataset from Kaggle. The dataset consists of labeled chest X-ray images categorized into **Normal** and **Pneumonia** classes. The data is organized into training, validation, and test sets, following an approximate **70/20/10 split**.

### Dataset Split Summary

| Dataset Split | NORMAL | PNEUMONIA |
|--------------|--------|-----------|
| **Training** | 1,109  | 3,016     |
| **Validation** | 317  | 861       |
| **Test** | 159  | 432       |

After loading the dataset into Python, we verified the directory structure and confirmed that all images were successfully loaded. Sample images from both classes were displayed to visually ensure correctness.

---

## 2. Data Cleaning

Since the dataset consists entirely of image files, data cleaning focused on detecting corrupted, duplicate, or irrelevant images. Each image was tested using image-reading functions to ensure it could be opened correctly.

- No corrupted images were found  
- All images belonged to either the **Normal** or **Pneumonia** class  
- No duplicate images were detected  

The dataset was therefore suitable for further analysis without modification.

---

## 3. Exploratory Data Analysis (EDA)

Exploratory data analysis was performed to better understand the dataset:

- Class distribution plots revealed a **significant class imbalance**, with Pneumonia cases outnumbering Normal cases
- Sample image visualization showed clearer lung regions in Normal images and cloudy or opaque regions in Pneumonia images
- Image dimension analysis showed varying sizes, motivating image resizing
- Pixel intensity histograms supported normalization to improve model training

---

## 4. Data Preprocessing

The following preprocessing steps were applied:

- Images resized to **224 × 224** pixels  
- Pixel values normalized to **[0, 1]**  
- Data augmentation techniques applied:
  - Random rotations  
  - Width and height shifts  
  - Zooming  
  - Horizontal flipping  
- Batch size set to **32**

To address class imbalance, **class weights** were computed:
- Normal: **1.86**
- Pneumonia: **0.68**

This ensured fairer learning by penalizing misclassification of minority class samples more heavily.

---

## 5. Modeling

All models were implemented and trained using the **TensorFlow deep learning framework with the Keras API**. Identical training, validation, and test splits were used across all experiments to ensure fair comparison.

### Baseline Models

- **Artificial Neural Network (ANN):**  
  A fully connected network with dropout layers. The ANN performed poorly on validation and test data, confirming that it is not well suited for image-based tasks due to lack of spatial feature extraction.

- **Custom Convolutional Neural Network (CNN):**  
  Included multiple convolutional and max-pooling layers followed by dense layers with dropout. The CNN significantly outperformed the ANN but was still inferior to transfer learning models.

### Transfer Learning Models

#### ResNet50 (10 Epochs)
- Pretrained on ImageNet
- Frozen convolutional base
- Classification head:
  - Global Average Pooling  
  - Dense (128 units, ReLU)  
  - Dropout (0.5)  
  - Sigmoid output layer  
- Optimizer: Adam (learning rate = 1e-4)

ResNet50 showed moderate accuracy and higher recall for Pneumonia cases but struggled to correctly classify Normal images.

#### DenseNet121 (10 Epochs)
- Same training setup as ResNet50
- Classification head:
  - Global Average Pooling  
  - Dense (128 units, ReLU)  
  - Dropout (0.5)  
  - Additional Dropout (0.3)  
  - Sigmoid output layer  

DenseNet121 achieved higher accuracy and better generalization due to dense connectivity and improved gradient flow.

---

## 6. Extended Training and Hyperparameter Tuning

Both ResNet50 and DenseNet121 were trained for **20 epochs**, and hyperparameter tuning was performed using **Keras Tuner**.

### Best ResNet50 Configuration
- Dense units: 256  
- Dropout: 0.3  
- Learning rate: 0.001  

### Best DenseNet121 Configuration
- Dense units: 128  
- Dropout: 0.6  
- Learning rate: 0.001  

The tuned DenseNet121 achieved the best performance:
- **Test Accuracy:** 92.05%  
- **Test Loss:** 0.2353  

---

## 7. Evaluation Metrics

To account for class imbalance and clinical relevance, multiple evaluation metrics were used:

- Accuracy and loss  
- Confusion matrices  
- Precision, recall, and F1-score  
- ROC curves and AUC  
- Precision–Recall curves and Average Precision (AP)  

DenseNet121 consistently produced fewer false negatives and achieved higher recall for Pneumonia cases, which is critical in medical diagnosis.

---

## 8. Conclusion

This project implemented a complete deep learning pipeline for pneumonia detection from chest X-ray images. After data cleaning, EDA, preprocessing, and model development, **DenseNet121** emerged as the best-performing model, achieving a test accuracy of **92.05%**.

The results demonstrate the effectiveness of **TensorFlow-based transfer learning** for medical image classification and highlight the importance of preprocessing, augmentation, and model selection.

---

## 9. Future Work

Future improvements may include:
- Fine-tuning additional layers of DenseNet121  
- Expanding to larger and more diverse datasets  
- Exploring ensemble methods  
- Deploying the model in clinical settings  
- Integrating explainable AI techniques such as **Grad-CAM**

---

## 10. References

Mooney, P. (2018). *Chest X-ray images (pneumonia)*. Kaggle.  
https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
