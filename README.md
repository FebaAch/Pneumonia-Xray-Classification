Machine_learning_final

Battle of the Networks: Comparing ResNet50 and DenseNet121 for Pneumonia Detection from Chest X-Rays
Group Members: Feba Achankunju and Jasmeen Kaur
1. Loading the Data
For this project, we used the Chest X-Ray (Pneumonia) dataset from Kaggle, which contains labeled chest X-ray images categorized into Normal and Pneumonia. The dataset is organized into three folders: training, validation, and testing, which follow an approximate 70/20/10 split.
| Dataset Split  | NORMAL | PNEUMONIA |
| -------------- | ------ | --------- |
| **Training**   | 1,109  | 3,016     |
| **Validation** | 317    | 861       |
| **Test**       | 159    | 432       |


After loading the dataset into Python, we displayed a snapshot of the directory structure and verified that the images loaded correctly. We also printed the number of images in each class for every split. The training set contains 1109 Normal images and 3016 Pneumonia images, the validation set contains 317 Normal and 861 Pneumonia images, and the test set contains 159 Normal and 432 Pneumonia images. Sample images from both classes were displayed to visually confirm that the data was being read properly.


2. Data Cleaning
Since this dataset consists entirely of image files, data cleaning mainly focused on checking for corrupted, duplicate, or irrelevant images. Each image file was tested using image-reading functions to confirm that it could be opened correctly. The results showed that no corrupted images were found in any of the dataset splits.
Additionally, all images were confirmed to belong to either the Normal or Pneumonia class, so no irrelevant files were present. Because each image file had a unique name and appeared only once, no duplicate images were detected. Based on these checks, the dataset was clean and suitable for further analysis and modeling without needing to remove or modify any files.

3. Exploratory Data Analysis (EDA)
To better understand the dataset, we performed several exploratory data analysis (EDA) steps using visualizations and summary statistics. First, we created bar plots of the class distribution for the training, validation, and test sets. These plots showed that the Pneumonia class is consistently larger than the Normal class across all three splits, confirming that the dataset is imbalanced.

Next, we displayed random sample images from both the Normal and Pneumonia classes. From these samples, we observed that Normal chest X-rays generally show clearer lung areas, while Pneumonia images commonly contain white patches and cloudy regions, which are typical signs of infection.

We also examined image dimensions using box plots, which showed that the original images vary in height and width. This confirmed the need to resize all images to a consistent size before modeling. Finally, we plotted a pixel intensity histogram, which showed that most pixel values are concentrated in the darker range with some bright areas. This supported our decision to normalize the pixel values during preprocessing to improve model learning.

4. Data Preprocessing
To prepare the dataset for modeling, several preprocessing steps were applied. First, all images were resized to 224 a 224 pixels so that they would be compatible with the ResNet50 and DenseNet121 models. Next, pixel values were normalized to a range of 0 to 1 by dividing by 255. This helps the neural networks train more efficiently and improves numerical stability.

To reduce overfitting and increase the variety of training data, we applied data augmentation techniques, including random rotations, width and height shifts, zooming, and horizontal flipping. These transformations create different variations of the same image during training and allow the models to generalize better. The batch size was set to 32 to balance memory usage and training speed.

Because the dataset is imbalanced, with significantly more Pneumonia images than Normal images, we calculated and applied class weights to correct this issue. The resulting class weights were approximately 1.86 for the Normal class and 0.68 for the Pneumonia class. Applying these weights ensured that misclassifying Normal images would be penalized more heavily during training, helping reduce bias toward the Pneumonia class.

5. Modeling
At this stage of the project, multiple deep learning models were developed and evaluated for chest X-ray–based pneumonia detection, including a baseline custom CNN, an ANN, and two transfer learning architectures: ResNet50 and DenseNet121. All experiments used the same training, validation, and test splits to ensure a fair comparison across models.
Baseline Models: ANN and Custom CNN
As baselines, an Artificial Neural Network (ANN) and a custom Convolutional Neural Network (CNN) were trained from scratch.
The ANN consisted of a Flatten layer followed by fully connected layers with dropout. While it achieved reasonable training performance, it struggled on validation and test data, confirming that ANN models are not well suited for image data because they fail to capture spatial features.
The custom CNN included multiple convolutional and max-pooling layers followed by dense layers with dropout. Compared to the ANN, the CNN performed substantially better, demonstrating the importance of spatial feature extraction for medical images. However, its performance remained inferior to transfer learning models, motivating the use of pretrained architectures.
Transfer Learning Models
ResNet50 (10 Epochs)
For ResNet50, ImageNet-pretrained weights were used with the convolutional base fully frozen, allowing the network to act as a fixed feature extractor. A custom classification head was added consisting of:
Global Average Pooling
Dense layer with 128 units and ReLU activation
Dropout (0.5)
Sigmoid output layer for binary classification
The model was trained for 10 epochs using the Adam optimizer (learning rate = 1e-4), binary cross-entropy loss, class weights, EarlyStopping, and ReduceLROnPlateau.
On the test set, ResNet50 achieved moderate accuracy and demonstrated higher recall for Pneumonia cases than for Normal cases, indicating a tendency to favor detection of infected lungs.
DenseNet121 (10 Epochs)
DenseNet121 was trained using the same transfer learning strategy, optimizer, loss function, callbacks, and class weights to maintain experimental consistency. Its classification head included:
Global Average Pooling
Dropout (0.5)
Dense layer with 128 units (ReLU)
Additional Dropout (0.3)
Sigmoid output layer
Due to its dense connectivity, DenseNet121 enables feature reuse across layers and improves gradient flow. As a result, it achieved higher test accuracy and lower loss than ResNet50, demonstrating stronger generalization and better handling of class imbalance.
Extended Training and Hyperparameter Tuning (20 Epochs)
To further improve performance, both ResNet50 and DenseNet121 were trained for 20 epochs, and hyperparameter tuning was performed using Keras Tuner.
For ResNet50, tuning focused on:
Number of dense units (64, 128, 256)
Dropout rate (0.3–0.7)
Learning rate (1e-2, 1e-3, 1e-4)
The best ResNet50 configuration used 256 units, 0.3 dropout, and a learning rate of 0.001, resulting in improved validation and test performance compared to the baseline ResNet model.
For DenseNet121, a similar tuning strategy was applied. The best configuration used:
128 dense units
Dropout rate of 0.6
Learning rate of 0.001
When trained for 20 epochs, the tuned DenseNet121 achieved a test accuracy of 92.05% and a test loss of 0.2353, outperforming all other models in the study.

8. Evaluation Metrics
To comprehensively evaluate the performance of the proposed models for pneumonia detection, multiple evaluation metrics were used. Given the class imbalance in the dataset and the medical importance of correctly identifying Pneumonia cases, accuracy alone is insufficient. Therefore, additional metrics such as precision, recall, F1-score, confusion matrices, ROC curves, and Precision–Recall curves were analyzed.
Accuracy and Loss
Test accuracy and test loss were used as initial indicators of overall model performance. Accuracy represents the proportion of correctly classified chest X-ray images, while loss reflects how well the predicted probabilities align with the true labels.
Among all evaluated models, the tuned DenseNet121 trained for 20 epochs achieved the highest performance, with a test accuracy of 92.05% and a test loss of 0.2353, outperforming ResNet50 and all baseline models. While ResNet50 showed strong improvements after hyperparameter tuning and extended training, its performance remained slightly lower than DenseNet121.


Confusion Matrix
Confusion matrices were generated for each model to analyze classification behavior in detail. These matrices report the number of:
True Positives (correctly detected Pneumonia cases),
True Negatives (correctly identified Normal cases),
False Positives, and
False Negatives.
The confusion matrices revealed that DenseNet121 produced fewer false negatives compared to ResNet50, indicating better sensitivity to Pneumonia cases. This is especially important in a clinical setting, where failing to detect pneumonia can have serious consequences.



Precision, Recall, and F1-Score
Classification reports were computed for all models, providing precision, recall, and F1-score for both Normal and Pneumonia classes.
Precision measures how many predicted Pneumonia cases were truly Pneumonia.
Recall (Sensitivity) measures how many actual Pneumonia cases were correctly identified.
F1-score balances precision and recall, making it particularly useful for imbalanced 
Datasets.
DenseNet121 demonstrated higher recall and F1-score for the Pneumonia class, indicating that it was more effective at detecting infected lungs while maintaining reliable predictions.
ROC Curve and AUC
	ROC curves and AUC scores were computed using test-set predictions. While both models showed strong discriminative ability, DenseNet121 achieved a higher AUC, demonstrating superior overall performance in distinguishing Pneumonia from Normal cases.

Precision–Recall Curve
	Precision–Recall curves and Average Precision (AP) scores were also evaluated. DenseNet121 achieved higher average precision, confirming more consistent and dependable Pneumonia detection across varying classification thresholds.


7. Conclusion 
In the final phase of the project, we completed the full pipeline for pneumonia detection using chest X-ray images, including data cleaning, exploratory data analysis (EDA), preprocessing, model development, and evaluation. The dataset was confirmed to be clean and free of corrupted images, though a class imbalance between Normal and Pneumonia cases was observed. EDA provided insights into image characteristics, guiding preprocessing decisions such as resizing, normalization, and augmentation.
We applied transfer learning using ResNet50 and DenseNet121 and also experimented with a custom CNN and a fully connected ANN. All models were trained with class weighting to address data imbalance. DenseNet121 consistently achieved the best performance, reaching a test accuracy of 92.05%, outperforming ResNet50 and the other models. Evaluation metrics including precision, recall, F1-score, ROC-AUC, and precision-recall curves confirmed DenseNet121’s superior ability to detect Pneumonia cases while maintaining reliable predictions for Normal images.
These results demonstrate that transfer learning with DenseNet121 is highly effective for pneumonia detection. The project highlights the importance of preprocessing, data augmentation, and careful model selection. Overall, the final outcomes provide a robust and reliable pneumonia detection system suitable for clinical or research applications.
Future Work: Further improvements can be explored by fine-tuning additional layers of DenseNet121, incorporating larger and more diverse datasets, and experimenting with ensemble models to enhance prediction accuracy. Deployment of the system in real-world clinical settings could also be considered, along with integration of explainable AI techniques like GRAD CAM to increase model interpretability for healthcare professionals.

8. References
Mooney, P. (2018, March 24). Chest X-ray images (pneumonia). Kaggle. https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia 
