# Plant Disease Detection Models using VGG16

## Overview
This repository contains two models for the detection of plant diseases using deep learning techniques, specifically leveraging the VGG16 architecture. The dataset used for training the models is the **Tomato-Village** dataset, which can be found on Kaggle. The dataset is focused on plant disease classification, and in this project, the **Variant-a (Multiclass classification)** subdirectory is used to train the models.

- [Tomato-Village dataset on Kaggle](https://www.kaggle.com/datasets/puneet6060/intel-image-classification)

## Models

### 1. **Model 1: `model.ipynb` (VGG16-based model for Multiclass Classification)**
This model is based on the VGG16 architecture, which is fine-tuned to classify plant diseases into multiple categories. The model includes several pre-processing techniques and data augmentations, such as image rotation, zoom, width, and height shifts to enhance training. A VGG16 model with a custom classifier on top was trained, and the first 15 layers of VGG16 were frozen to retain the learned features, with the remaining layers being trainable for fine-tuning.

- **Accuracy:** 62%

#### Description:
The model was trained using the **Variant-a (Multiclass classification)** subdirectory of the Tomato-Village dataset. It aims to detect various plant diseases using images. The model performed moderately well with an accuracy of 62% during evaluation.

---

### 2. **Model 2: `model2.ipynb` (Transfer Learning-based Model using VGG16)**
This model uses a transfer learning approach, starting with the pre-trained VGG16 model (trained on ImageNet) and adding custom layers for the disease classification task. The base VGG16 model's layers are frozen, and a new set of layers is added on top for multiclass classification. The model was trained on a dataset containing labeled plant disease images and was evaluated using both training and validation sets.

- **Accuracy:** 86%

#### Description:
Model 2 utilizes transfer learning by fine-tuning the VGG16 model to detect plant diseases from images. The custom layers are designed to predict multiple classes (disease categories) based on the extracted features from the pre-trained VGG16 model. This method significantly outperformed the first model, achieving an accuracy of 86%, making it better suited for the plant disease detection task.

---

## Dataset
The dataset used for training the models is the **Tomato-Village dataset** from Kaggle. It consists of various categories of tomato plant diseases. For this project, only the **Variant-a (Multiclass classification)** subdirectory was used.

## Modules Used

- **NumPy**: Used for numerical operations and array manipulations.
- **Pandas**: Used for data handling and manipulation.
- **Matplotlib**: Used for plotting and visualizing training results such as accuracy and loss graphs.
- **Seaborn**: Used for data visualization (especially for the visualizations of confusion matrix and data distribution).
- **TensorFlow**: Deep learning framework used for creating, training, and evaluating neural networks.
  - **Keras** (part of TensorFlow): Used to implement the VGG16 architecture, transfer learning, data augmentation, and training pipelines.
  - **VGG16**: Pre-trained convolutional neural network used as the base model for transfer learning.
- **OpenCV**: Optional, for image pre-processing tasks.
- **OS**: For handling file paths and directories.
- **tf.keras.callbacks.ModelCheckpoint**: Used to save the best model during training.

## Results Comparison

- **Model 1 (VGG16-based model)** achieved an accuracy of 62%.
- **Model 2 (Transfer learning-based VGG16)** achieved an accuracy of 86%.

Model 2 provides better results and is more suitable for this dataset due to the benefits of transfer learning. The pre-trained VGG16 model allows the network to leverage previously learned features, resulting in improved performance for the plant disease classification task.

## Conclusion
The **Model 2** is the better-performing model, and it is recommended for use in real-world applications based on its higher accuracy of 86% (Can be increased if trained for more epochs [around 15-20]). Model 1 can still serve as a baseline but does not perform as well as Model 2 on this dataset.

