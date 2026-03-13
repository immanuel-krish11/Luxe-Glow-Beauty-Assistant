# import this file and call predict_skin_type(image_path) to get direct result
# the function returns the skin_type [dry/normal/oily] along with accuracy of the predicted output

# importing al necessary modules
import numpy as np
import pandas as pd
import os
import torch 
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

import torch.nn.functional as F

# setting some default values for the model
MODEL_PATH = "models/skin_type_efficientnetb0_acc88.pth"
IMG_SIZE = 224
OUT_CLASSES = 3
index_label = {0: "dry", 1: "normal", 2: "oily"}
device = "cuda" if torch.cuda.is_available() else "cpu"


# Recreating the model architecture
def load_trained_model(model_path):
    # Load the base EfficientNet-B0
    model = models.efficientnet_b0(weights=None)
    
    # Re-modify the classifier head to match 3 classes
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, OUT_CLASSES)
    
    # Load the saved state_dict
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval() # Crucial: set to evaluation mode
    return model


# define prediction function and preprocessing the image
def predict_skin_type(model, image_path): # remove model if you want to use the line below which loads trained model here
    # model = load_trained_model(MODEL_PATH)        # call in the main file or here
    # Standard validation transforms
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Open image and convert to array to avoid the 'PIL.Image' error you saw earlier
    img = Image.open(image_path).convert("RGB")
    img_array = np.array(img)
    
    # Preprocess and add batch dimension
    img_tensor = transform(img_array).unsqueeze(0).to(device)

    with torch.no_grad(): # Disable gradients for faster inference
        outputs = model(img_tensor)
        # Calculate probabilities using Softmax
        probs = F.softmax(outputs, dim=1)
        
        # Get highest probability and its index
        conf, predicted = torch.max(probs, 1)
        
    accuracy_pct = conf.item() * 100
    label_name = index_label[predicted.item()]
        
    return index_label[predicted.item()], accuracy_pct