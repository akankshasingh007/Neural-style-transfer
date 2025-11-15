# Neural Style Transfer

A simple deep-learning project that blends the **content** of one image with the **style** of another using **Neural Style Transfer (NST)**. The model uses **VGG19** to extract features and optimizes a new image that matches the content structure while adopting the artistic style.

---

## Overview

This project:
- Takes a **content image** (photo)
- Takes a **style image** (artwork)
- Generates a new image combining both

The process uses feature extraction, Gram Matrices, and a custom loss function to guide the optimization.

---

## What You’ll Learn

- How Convolutional Neural Networks handle image features  
- What a feature extractor is and how VGG19 is used  
- How Gram Matrices represent style  
- How content loss and style loss are calculated  
- How to build a custom training loop using TensorFlow/Keras  

---

## Tools & Libraries

- Python  
- TensorFlow / Keras  
- NumPy  
- Pillow (PIL)  
- Streamlit  

---

## Features

- Upload a content image  
- Upload a style image  
- Generate a stylized output  
- Run everything inside a simple Streamlit UI  

---

## How to Run

```bash
streamlit run app.py
