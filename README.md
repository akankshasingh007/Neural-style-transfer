 Neural Style Transfer Project
 Overview
This project combines two images — one for content (like a photo) and one for style (like a painting) — to create a new image that looks like the photo but painted in the style of the artwork.
We’ll use deep learning to do this by using a pre-trained model (VGG19) to extract and mix the features of both images.

 What I’ll Learn
How Convolutional Neural Networks (CNNs) can be used for image processing.
What a feature extractor is and how it helps get content and style from an image.
What a Gram Matrix is and how it represents the texture and color patterns (the “style”).
How to use content loss and style loss to generate new images.
How to build a custom training loop using TensorFlow and Keras.

 Tools & Libraries
Python
TensorFlow / Keras
NumPy
Pillow (PIL)
Streamlit – for the web app where users can upload their images

 Goal
By the end, the app should:
Let users upload a content image and a style image.
Generate and display a new stylized image.
