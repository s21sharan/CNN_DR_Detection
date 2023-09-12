
# CNN Model for Diabetic Retinopathy Detection
Sharan Subramanian, 9/11/2023

Materials for Research Paper, *Convolutional Neural Network Model for Diabetic Retinopathy Feature Extraction and Classification*
## Description
This code provides a deep learning solution to diagnose diabetic retinopathy from fundus images. Diabetic retinopathy is an ocular manifestation of diabetes that affects blood vessels in the retina. The code uses a convolutional neural network (CNN) to classify images into various stages of the disease.

## Requirements
- Python 3.x
- numpy
- pandas
- os
- random
- sys
- cv2
- matplotlib
- keras
- tensorflow
- sklearn
- google.colab (if running on Google Colab)

## Steps to Use
1. Mount Google Drive: If you're running this code on Google Colab, it begins by mounting your Google Drive to access data files.

2. Preprocessing: The code reads images and scales them to the specified width and height (128x128 in this code). Images are then normalized by dividing pixel values by 255.

3. Data Mapping: The 'AnnotationsOfTheClassifications.csv' file provides labels for each image. The code maps these labels to respective images.

4. Data Augmentation: The code uses Keras's ImageDataGenerator to perform data augmentation. This process involves making slight modifications (rotations, zooms, shifts) to the original images to increase the size of the training dataset and prevent overfitting.

5. Model Definition: A CNN model is defined using Keras with three convolution blocks followed by a dense layer. The createModel function defines this architecture.

6. Training: The model is trained on the preprocessed dataset with the specified number of epochs, learning rate, and batch size.

7. Evaluation: After training, the model's performance is evaluated on a validation set.

8. Visualization: A plot depicting the training loss, validation loss, training accuracy, and validation accuracy over epochs is generated and saved to the drive.

## Functions
`classes_to_int()`: Convert class labels into integers.

`int_to_classes()`: Convert integers back into class labels.

`preprocessTrainData()`: Preprocess training images.

`readTrainCsv()`: Read the CSV file that contains image labels.

`createModel()`: Define the CNN model architecture.

`evaluate_model()`: Evaluate the trained model on a given test set.

## Results
At the end of training, the model is saved to your Google Drive. Additionally, a plot displaying the loss and accuracy over epochs is saved. The model's performance on the training set is also displayed in a table.
## Notes
- Ensure that your Google Drive contains the necessary dataset and the CSV file for annotations.
- Make sure to adjust paths accordingly if not using Google Colab or if the dataset is stored in a different location.
- You can adjust the WIDTH, HEIGHT, EPOCHS, INIT_LR, and BS variables to change the image dimensions, number of epochs, learning rate, and batch size respectively.

