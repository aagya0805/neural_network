# neural_network
# Multimodal Sentiment Analysis (DistilBERT + ResNet50)

This repository contains the implementation of a Late-Fusion Multimodal Neural Network designed to classify sentiment into three categories: Positive, Neutral, and Negative, by integrating textual and visual data.

## Project Overview
In digital communication, text alone often fails to capture nuances like sarcasm. This project implements a dual-stream architecture:
- **Text Branch:** Utilizes **DistilBERT** for context-sensitive embeddings.
- **Visual Branch:** Utilizes **ResNet50/MobileNetV2** for high-level spatial feature extraction.
- **Fusion:** A late-fusion layer concatenates these features to produce a final sentiment prediction.

## Major Technical Outcomes
- **Performance Boost:** The multimodal model significantly outperformed the unimodal (text-only) baseline, particularly in identifying sarcasm where text and images provided conflicting signals.
- **Optimization:** Implemented using the Adam optimizer (LR=0.0001) and Sparse Categorical Cross-Entropy loss.
- **Robustness:** Achieved stable convergence over 20 epochs with stratified data partitioning to handle class imbalance.
- **Inference Interface:** Integrated with **Gradio** for real-time sentiment prediction.

## Inference Instructions
A ready-to-use model file (`multimodal_sentiment_model.pth`) is included in this repository. Follow these steps to perform inference on new image-text pairs.

### 1. Prerequisites
Ensure you have the required libraries installed:
```bash
pip install torch torchvision transformers pillow
