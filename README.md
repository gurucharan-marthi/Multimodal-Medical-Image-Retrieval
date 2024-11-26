# Multimodal Medical Image Retrieval System for Clinical Decision Support

This repository contains the code implementation for the **Multimodal Medical Image Retrieval System for Clinical Decision Support** paper, authored by Gurucharan Marthi, Krishna Kumar, and S. Sidtharth. The system aims to assist clinical decision-making by retrieving relevant 2D and 3D medical images from databases using multimodal content-based image retrieval (CBMIR) techniques.

You can read the full paper here: [Multimodal Medical Image Retrieval System for Clinical Decision Support](https://doi.org/10.1016/B978-0-443-15452-2.00025-X)

## Objective

This system aims to bridge the gap between manual annotations and automated image retrieval, using deep learning methods that combine Convolutional Neural Networks (CNNs) with Autoencoders. The system retrieves multimodal, multi-anatomy images (e.g., CT, MRI, X-rays) to aid in clinical diagnosis and decision-making.

## Overview of Models

This repository includes the implementation of five different models for medical image retrieval:

1. **LeNetCoder**  
   A hybrid model combining the LeNet architecture (CNN) with an Autoencoder, consisting of 14 layers (7 in the encoder, 7 in the decoder). The encoder extracts features using convolution and pooling, while the decoder reconstructs the images.

2. **VGGCoder**  
   A hybrid CNN + Autoencoder architecture based on VGG16. This model has 28 layers with 14 weight layers, including convolutional layers, pooling layers, and fully connected layers. It processes 3D input images and reduces the image dimensions for efficient retrieval.

3. **Noisy VGGCoder**  
   An enhancement to the VGGCoder that includes a denoising autoencoder. By adding noise (Random, Gaussian, or Salt-and-Pepper) to the input images, this model learns to reconstruct clean images and thus avoids learning the identity function.

4. **LSTM VGGCoder**  
   This model adds a Long Short-Term Memory (LSTM) layer on top of the VGG16-based architecture. It learns temporal dependencies in the image data, improving the model’s performance on sequential or time-dependent image retrieval tasks.

5. **ResCoder**  
   A deep model built using 80 layers, incorporating residual blocks to address the vanishing gradient problem. This model leverages skip connections to ensure that deeper networks retain gradient flow, improving retrieval accuracy.

## Dataset

The dataset used in the paper consists of medical images from various modalities:

- **2D Images:** 27,200 images across 11 classes (CT, MRI, X-rays, etc.)
- **3D Images:** 17,100 images across 9 classes (MRI modalities, CT scans)
- **Sources:** Open repositories such as ADNI, TCIA, and Radiopaedia.

### Preprocessing
- All images are resized to 256x256 for 2D and 64x64x64 for 3D images.
- Images are converted to grayscale and augmented to increase the dataset size.
- Models are evaluated using **Precision**, **Recall**, **F-score**, **MAP**, and **Retrieval Time**.

## Repository Structure


## Requirements

- Python 3.x
- TensorFlow
- Keras
- Numpy
- Scikit-learn
- Matplotlib
- OpenCV
- CUDA (for GPU support)

## Evaluation

The models are evaluated using the following metrics:

- MAP (Mean Average Precision)
- Dice Score
- Jaccard Index
- MSE (Mean Squared Error)
- Retrieval Time

##  Performance Comparison
Comparison tables evaluate the performance of each model based on Precision (P), Recall (R), and F-score (F). From these comparisons, it is observed that Noisy VGGCoder and ResCoder perform well for both 2D and 3D image retrieval with a scope size of 20.

### MAP Comparative Analysis:

| Model               | MAP (2D) | MAP (3D) |
|---------------------|----------|----------|
| LeNet Coder         | 0.93     | 0.85     |
| VGG Coder           | 0.95     | 0.88     |
| Noisy VGG Coder     | 0.96     | 0.90     |
| LSTM VGG Coder      | 0.93     | 0.89     |
| Res Coder           | 0.96     | 0.92     |

These results demonstrate the efficacy of the proposed models in retrieving accurate images across both 2D and 3D datasets.

## Key Results

- **Best Performance**: The **ResCoder** model achieves the highest performance with a MAP of 96% for 2D images and 92% for 3D images.
- **Noisy VGGCoder** performed closely behind, showing the importance of denoising for improved retrieval performance.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


