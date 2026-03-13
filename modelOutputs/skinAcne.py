# import this file and call predict_skin_acne(image_path) to get direct result
# the function returns the skin_acne [Mild/Moderate/Severe/Very Severe] along with accuracy of the predicted output

"""
=============================================================================
Acne Severity Classification: Phase 2 - Independent Inference Module
Framework: PyTorch
Target Use Case: Production deployment, API backend, or personal project usage.
=============================================================================
This script operates independently of the training data. It reconstructs the 
neural network topology, injects the serialized weights, and executes 
highly-optimized forward passes on individual target images.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image

class AcneSeverityPredictor:
    """
    A modular predictor class encapsulating the neural network topology,
    parameter state, and required data preprocessing protocols.
    """
    
    def __init__(self, model_path, num_classes=4):
        """
        Initializes the inference engine by rebuilding the model graph and
        loading the serialized state dictionary.
        """
        # map_location ensures graceful degradation to CPU if the model was 
        # trained on a Kaggle GPU but is being executed on a local machine.
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.class_names = ['Mild', 'Moderate', 'Severe', 'Very Severe']
        
        # 1. Reconstruct the precise network topology used during training
        # pretrained=False prevents the unnecessary downloading of ImageNet weights
        print(" Reconstructing ResNet-50 topological graph...")
        self.model = models.resnet50(pretrained=False) 
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)
        
        # 2. Deserialize and inject the optimized parameter state
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model artifact not found at {model_path}")
            
        print(f" Ingesting optimized parameters from {model_path}...")
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        
        # 3. Secure the model for inference operations
        self.model = self.model.to(self.device)
        self.model.eval() # CRITICAL: Disables training-specific regularizations
        
        # 4. Define the strict inference-time tensor transformations
        # Notice the absence of stochastic augmentations (no flips or crops)
        self.transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet mean
                             std=[0.229, 0.224, 0.225])   # ImageNet std
        ])
    

    def predict(self, image_path):
        """
        Executes a streamlined forward pass on a singular image file, 
        extracting the maximum likelihood class and probability distribution.
        """
        try:
            # Ingest image and enforce RGB channel configuration
            image = Image.open(image_path).convert('RGB')
            
            # Apply preprocessing and append a batch dimension: shape becomes (1, C, H, W)
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Context Manager: Temporarily dismantle the autograd engine
            # This drastically reduces VRAM overhead and accelerates inference latency
            with torch.no_grad():
                # Forward pass: Project the tensor through the network
                raw_logits = self.model(input_tensor)
                
                # Apply Softmax to translate raw, unbounded logits into a 0-1 probability curve
                probabilities = F.softmax(raw_logits, dim=1)
                
                # Extract the peak probability and its corresponding structural index
                confidence, predicted_idx = torch.max(probabilities, 1)
                
            predicted_class = self.class_names[predicted_idx.item()]
            confidence_score = confidence.item() * 100
            
            return predicted_class, confidence_score
            
        except Exception as e:
            print(f" Inference failed on target {image_path}: {e}")
            return None, None




# ---------------------------------------------------------------------------
# Execution Block: Sample Usage Simulation
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Define artifact and target pathways
    # In Kaggle, the model is retrieved from the /working/ output of Phase 1
    SAVED_MODEL_PATH = "/Users/krishprakash/Desktop/skin-care-mk2/models/acne_resnet50_best.pth" 
    
    # Path to an unseen test image (e.g., uploaded by the user)
    SAMPLE_IMAGE_PATH = "/Users/krishprakash/Desktop/skin-care-mk2/custom-images/00048_png_jpg.rf.2cd59aace0c099f72e3fad3b12ebfc51.jpg" 
    
    if os.path.exists(SAVED_MODEL_PATH) and os.path.exists(SAMPLE_IMAGE_PATH):
        # Instantiate the deployment object
        predictor = AcneSeverityPredictor(model_path=SAVED_MODEL_PATH)
        
        # Trigger the diagnostic inference sequence
        prediction, confidence = predictor.predict(SAMPLE_IMAGE_PATH)
        
        if prediction:
            # Construct a formalized terminal report
            print("\n=============================================")
            print("        AI DIAGNOSTIC INFERENCE REPORT       ")
            print("=============================================")
            print(f"Target Image:      {os.path.basename(SAMPLE_IMAGE_PATH)}")
            print(f"Algorithm Output:  [{prediction.upper()}]")
            print(f"System Confidence: {confidence:.2f}%")
            print("---------------------------------------------")
    else:
        print(" Sample execution aborted. Please verify the file paths for the.pth model and the test image.")