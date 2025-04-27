
# Image-Based Scene Classification Using Deep Learning

### Problem Description 
Automated scene classification is crucial in computer vision, with applications in geospatial analysis, autonomous navigation, and content organisation. This project aims to develop a deep learning model capable of classifying images into six categories: buildings, forests, glaciers, mountains, sea, and streets. Manually sorting large image datasets is inefficient and prone to errors, making an automated system highly beneficial. A robust model must generalise well across diverse images because natural scenes exhibit high variability in lighting, perspective, and environmental conditions. This project uses Convolutional Neural Networks (CNNs) to effectively capture image spatial hierarchies and distinguish between different scene categories. 
**Input**: A high-resolution RGB image (150x150 pixels) in `.jpg` format. 
**Output**: Text representing the image as one of the six categories: building, forest, glacier, sea,  streets or mountain.

### Dataset 
##### Source: Intel image classification dataset 
Pre-labelled images of size 150x150 of natural scenes around the world, split into train, test and prediction. 
[Link:](https://www.kaggle.com/datasets/puneet6060/intel-image-classification)

##### Structure:  
The actual data_set contains 14000 images for training. I have split the training data into train and validation. After finishing train the model was evaluated on the test data and results are shown in the section- "Evaluation on Test data" below.
● Training Data: 11,200 images
● Validation Data: 2800 (split from the Training Data) 
● Test Data: 7,000 images 
● Prediction Data: 3,000 images

A sample [data](./data) directory has been added to the repository. It has six sub-directories each belonging to a class and contain ten sample images of that class.   

### Choice of Model 
To effectively address the classification task, a deep CNN architecture will be implemented. CNNs are ideal for image-based tasks due to their ability to detect and learn spatial hierarchies, local features, and global patterns. This project has potential applications in remote sensing, tourism, urban planning, and content-based image retrieval. The model’s high accuracy and efficiency will enable seamless integration into various real-world applications requiring automated scene recognition.

### Installation
###### Clone the repository
git clone https://github.com/anaivecoder/project_Achyusman_Samanta
cd project_Achyusman_Samanta

###### Install the required packages:
```bash
pip install torch torchvision
pip install scikit-learn
pip install matplotlib seaborn tqdm Pillow numpy
```
In your virtual python environment run `train.py` to train the model on the data_set (download it from the link above).
To use the model for predicting your images run `predict.py`. **It takes the path of a directory that contain your images as input and returns image name: predicted_label as the output.**

### Evaluation on the Test Data 
Final Test Accuracy: 0.9204
##### Classification Report on Test Set:

| Class      | Precision | Recall | F1-Score | Support |
|------------|-----------|--------|----------|---------|
| Buildings  | 0.87      | 0.94   | 0.91     | 424     |
| Forest     | 0.97      | 1.00   | 0.98     | 450     |
| Glacier    | 0.89      | 0.90   | 0.89     | 507     |
| Mountain   | 0.89      | 0.89   | 0.89     | 482     |
| Sea        | 0.95      | 0.94   | 0.94     | 478     |
| Street     | 0.96      | 0.86   | 0.91     | 447     |

| Metric          | Precision | Recall | F1-Score | Support |
|-----------------|-----------|--------|----------|---------|
| Accuracy        | -         | -      | 0.92     | 2788    |
| Macro Average   | 0.92      | 0.92   | 0.92     | 2788    |
| Weighted Average| 0.92      | 0.92   | 0.92     | 2788    |

Accuracy measures the overall correctness of the model by calculating the ratio of correctly predicted images to the total number of images. Since the Intel dataset has well-balanced classes (buildings, forest, glacier, mountain, sea, street) with similar numbers of samples, accuracy serves as a reliable and intuitive performance indicator.

However, to ensure that the model performs well across all classes individually, we also consider the F1-Score. The F1-Score is the harmonic mean of precision and recall, helping to balance false positives and false negatives. It is particularly useful for highlighting any class imbalance issues or cases where the model performs better on some classes than others.









