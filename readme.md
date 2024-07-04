# Deep Learning Image Classification using Transfer Learning

This project focuses on deep learning image classification using transfer learning. We utilize the EfficientNetB0 model for feature extraction and fine-tuning to enhance the model's performance.

## Project Overview

- **Model**: EfficientNetB0
- **Techniques**: Feature extraction and fine-tuning
- **Training Data**: 750 images per class from 10 classes (total of 7500 images)
- **Test Data**: 250 images per class from 10 classes (total of 2500 images)
- **Evaluation**: Analyzing incorrect predictions using a Seaborn heatmap and visualizing high-confidence incorrect predictions
- **Deployment**: Streamlit app for user-friendly image uploads and predictions

## Dataset
We get the dataset images from the below link 
```
https://storage.googleapis.com/ztm_tf_course/food_vision/10_food_classes_all_data.zip
```

## Model Training

We first use EfficientNetB0 for feature extraction and then fine-tune the model to improve performance. The training accuracy reached 97%, while the test accuracy was around 93%.

## Error Analysis

To further improve the model, we analyzed incorrect predictions:
- **Heatmap**: A Seaborn heatmap was used to visualize the confusion matrix and understand where the model struggled.
- **High-Confidence Errors**: We visualized images with high-confidence incorrect predictions to identify patterns and potential areas for improvement.

## Streamlit App

We created a Streamlit app to allow users to upload their own images and get predictions from the trained model. The app provides an easy-to-use interface for image classification.

## How to Use

1. **Clone the Repository**
   ```bash
   git clone https://github.com/rajkumar/deep-learning-image-classification.git
   cd deep-learning-image-classification
2. **Install Dependencies**

    ```
    pip install -r requirements.txt
    Run the Streamlit App
    ```
3. **Streamlit interface**
    ```
    streamlit run app.py
    ```
4. **Upload Images** : Use the interface to upload your images and get predictions.

## Results
The model performed well on both the training and test datasets, with accuracy scores of 97% and 93%, respectively. Further error analysis and visualizations helped identify areas for potential improvement.
