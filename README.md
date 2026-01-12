# Machine Learning Final Project

## Battle of the Networks: Comparing ResNet50 and DenseNet121 for Pneumonia Detection from Chest X-Rays

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

- No corrupted images were found.
- All images belonged to either the **Normal** or **Pneumonia** class.
- No duplicate images were detected.

Based on these checks, the dataset was deemed clean and suitable for further analysis.

---

## 3. Exploratory Data Analysis (EDA)

Exploratory data analysis was performed to better understand the dataset characteristics:

- Bar plots of class distribution confirmed a **significant class imbalance**, with Pneumonia cases outnumbering Normal cases.
- Random sample images showed that Pneumonia X-rays commonly contain white or cloudy regions, while Normal X-rays exhibit clearer lung fields.
- Image dimension analysis revealed varying sizes, motivating resizing during preprocessing.
- Pixel intensity histograms showed most values clustered in darker ranges, supporting pixel normalization.

---

## 4. Data Preprocessing

The following preprocessing steps were applied:

- All images were resized to **224 × 224** pixels.
- Pixel values were normalized to the range **[0, 1]**.
- Data augmentation techniques were applied, including:
  - Random rotations
  - Width and height shifts
  - Zooming
  - Horizontal flipping
- Batch size was set to **32**.

To address class imbalance, **class weights** were computed:
- Normal: **1.86**
- Pneumonia: **0.68**

This ensured that misclassification of Normal cases was penalized more heavily during training.

---

## 5. Modeling

Multiple deep learning models were trained and evaluated using the same dataset splits for fair comparison.

### Baseline Models
- **Artificial Neural Network (ANN):**  
  Included fully connected layers with dropout. The ANN struggled with generalization, confirming that fully connected networks are not well suited for image data.
  
- **Custom CNN:**  
  Included convolutional and max-pooling layers followed by dense layers with dropout. The CNN performed significantly better than the ANN but was outperformed by transfer learning models.

### Transfer Learning Models

#### ResNet50 (10 Epochs)
- Pretrained on ImageNet
- Frozen convolutional base
- Classification head:
  - Global Average Pooling
  - Dense (128, ReLU)
  - Dropout (0.5)
  - Sigmoid output
- Optimizer: Adam (learning rate = 1e-4)

ResNet50 showed moderate accuracy and higher recall for Pneumonia cases but struggled with Normal images.

#### DenseNet121 (10 Epochs)
- Same training setup as ResNet50
- Classification head:
  - Global Average Pooling
  - Dense (128, ReLU)
  - Dropout (0.5)
  - Additional Dropout (0.3)
  - Sigmoid output

DenseNet121 achieved higher accuracy and better generalization due to dense connectivity and feature reuse.

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

The tuned DenseNet121 achieved the best performance with:
- **Test Accuracy:** 92.05%
- **Test Loss:** 0.2353

---

## 7. Evaluation Metrics

Given the class imbalance and clinical importance of detecting Pneumonia, multiple evaluation metrics were used:

- Accuracy and loss
- Confusion matrices
- Precision, recall, and F1-score
- ROC curves and AUC
- Precision–Recall curves and Average Precision (AP)

DenseNet121 consistently outperformed ResNet50, producing fewer false negatives and higher recall for Pneumonia cases, which is critical in medical diagnosis.

---

## 8. Conclusion

This project implemented a complete deep learning pipeline for pneumonia detection from chest X-ray images. After data cleaning, EDA, preprocessing, and model development, DenseNet121 emerged as the best-performing model, achieving a test accuracy of **92.05%**.

The results demonstrate the effectiveness of transfer learning for medical image classification and highlight the importance of proper preprocessing, data augmentation, and model selection.

---

## 9. Future Work

Future improvements could include:
- Fine-tuning additional layers of DenseNet121
- Using larger and more diverse datasets
- Exploring ensemble methods
- Deploying the model in real-world clinical settings
- Integrating explainable AI techniques such as **Grad-CAM** to improve interpretability

---

## 10. References

Mooney, P. (2018). *Chest X-ray images (pneumonia)*. Kaggle.  
https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
