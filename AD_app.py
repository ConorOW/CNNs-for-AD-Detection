#hide
import gradio as gr
from fastai.vision.all import *

# Load in our resnet18 AD learner
learn = load_learner('./AD-learner.pkl')

# Define a prediction function for our model
labels = learn.dls.vocab
def predict(img):
    img = PILImage.create(img)
    pred,pred_idx,probs = learn.predict(img)
    return {labels[i]: float(probs[i]) for i in range(len(labels))}

# Set some parameters for display
title = "Alemeimer's Disease Classifier"
description = "A Convolutional Neural Network trained to classify Alzheimer's Disease from brain MRI scans"

# Set our gradio Interface object with options
gr.Interface(fn=predict,
             title=title,
             description=description,
             inputs=gr.Image(),
             outputs=gr.Label(num_top_classes=2)).launch(share=True)