# this module returns the skin tone of a face of a person.
# this module returns monk skin tone from 1 - 10
# call predict_skin_tone by giving image path and it will return a monk skin tone

from typing import List
from PIL import Image
import numpy as np
from typing import Tuple
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
import os


def extract_skin_patches(
    image: Image.Image,
    mask: np.ndarray,
    patch_size: int = 224,
    stride: int = 112,
    purity_threshold: float = 0.85
) -> List[Image.Image]:
    """
    Traverses an image and its corresponding segmentation mask to extract
    localized patches that predominantly consist of valid skin pixels.

    Args:
        image (PIL.Image): The original input RGB image.
        mask (np.ndarray): Binary segmentation mask (1 for skin, 0 otherwise).
        patch_size (int): Spatial dimension of the extracted square patch.
        stride (int): Step size for the sliding window traversal algorithm.
        purity_threshold (float): Minimum required fraction of skin pixels in the patch.

    Returns:
        List[PIL.Image]: A collection of valid, cropped skin patches ready for classification.
    """
    img_array = np.array(image)
    h, w, _ = img_array.shape
    patches = []

    # Sliding window extraction across the image height and width
    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            # Extract the local sub-patch from the binary mask
            mask_patch = mask[y:y+patch_size, x:x+patch_size]

            # Calculate the density of skin pixels within this specific spatial window
            skin_ratio = np.sum(mask_patch) / (patch_size * patch_size)

            # Accept the image patch if it exceeds the strict purity constraint
            if skin_ratio >= purity_threshold:
                img_patch = img_array[y:y+patch_size, x:x+patch_size, :]
                patches.append(Image.fromarray(img_patch))

    # Algorithmic Fallback Mechanism:
    # If no patch meets the strict 85% purity (e.g., small faces or low resolution),
    # extract the global bounding box of the largest connected skin component.
    if not patches and np.sum(mask) > 0:
        y_indices, x_indices = np.where(mask == 1)
        y_min, y_max = y_indices.min(), y_indices.max()
        x_min, x_max = x_indices.min(), x_indices.max()

        # Crop to the global bounding box and forcefully resize to the required patch_size
        fallback_patch = image.crop((x_min, y_min, x_max, y_max))
        fallback_patch = fallback_patch.resize((patch_size, patch_size), Image.BILINEAR)
        patches.append(fallback_patch)

    return patches

def build_skin_tone_classifier(num_classes: int = 10) -> nn.Module:
    """
    Instantiates a pre-trained ResNet50 model and modifies the terminal
    fully connected layer for the specified number of target classes.
    """
    # Load the most accurate available ImageNet pre-trained weights
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    
    # Replace the classification head to map to the 10 MST categories
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    
    return model

def predict_patches(
    patches: List[Image.Image],
    model: nn.Module,
    device: torch.device
) -> Tuple[int, float]:
    """
    Executes batched inference over a collection of image patches, applying
    softmax and mean pooling to derive a consensus skin tone prediction.
    """
    if not patches:
        raise ValueError("No valid skin patches were extracted from the input image.")

    # Standard deterministic inference transforms matching the training state
    inference_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),  # convert PIL image to Tensor
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # typical ImageNet means
            std=[0.229, 0.224, 0.225]    # your standard deviations
        )
    ])


    # Process patches into a single batched tensor and move to the target hardware device
    tensor_batch = torch.stack([inference_transform(p) for p in patches]).to(device)

    # Ensure Dropout and BatchNorm layers are locked in evaluation mode
    model.eval()

    with torch.inference_mode():
        logits = model(tensor_batch)

        # Convert raw logits to probability distributions via the Softmax activation
        probabilities = torch.nn.functional.softmax(logits, dim=1)

        # Aggregate (Mean Pool) probabilities across all patches for the given face
        consensus_probs = probabilities.mean(dim=0)

        # Extract the maximum probabilistic confidence and its categorical index
        confidence_score, predicted_idx = torch.max(consensus_probs, dim=0)

    # Shift 0-indexed tensor output back to the standard 1-indexed Monk Skin Tone scale
    predicted_mst = predicted_idx.item() + 1
    confidence_percentage = confidence_score.item() * 100.0

    return predicted_mst, confidence_percentage

def predict_skin_tone(seg_model, seg_processor,image_path: str) -> Tuple[str, float]:
    """
    Top-level API: Reads an image, extracts face skin regions via semantic
    segmentation, runs the ResNet classifier on local patches, and aggregates
    predictions to return the final Monk Skin Tone label and confidence percentage.
    """
    #if you don't want to load model in app.py un-comment the following
    device = torch.device("cuda" if torch.cuda.is_available() else
    
                          "mps" if torch.backends.mps.is_available() else "cpu")

    # 1. Load the target image from the local filesystem
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        raise IOError(f"Failed to load image at {image_path}: {e}")

    # 2. Initialize the Semantic Segmentation (Face Parsing) framework
    # loading model in app.py
    # model_name = "jonathandinu/face-parsing"
    # seg_processor = SegformerImageProcessor.from_pretrained(model_name)
    # seg_model = SegformerForSemanticSegmentation.from_pretrained(model_name).to(device)
    # seg_model.eval()

    # 3. Generate pure binary skin mask utilizing the model
    inputs = seg_processor(images=image, return_tensors="pt").to(device)
    with torch.inference_mode():
        seg_outputs = seg_model(**inputs)

    upsampled_logits = F.interpolate(
        seg_outputs.logits, size=image.size[::-1], mode='bilinear', align_corners=False
    )
    semantic_map = upsampled_logits.argmax(dim=1).squeeze().cpu().numpy()
    skin_mask = (semantic_map == 1).astype(np.uint8)  # Index 1 is the 'skin' class

    # 4. Extract localized skin patches using the sliding window methodology
    patches = extract_skin_patches(image, skin_mask)

    # 5. Initialize the trained ResNet classifier
    classifier = build_skin_tone_classifier(num_classes=10)
    checkpoint_path = "/Users/krishprakash/Desktop/skin-care-mk2/models/best_model_skin_tone.pth.tar"

    if os.path.exists(checkpoint_path):
        # Load the saved state dictionary generated by the training loop
        checkpoint = torch.load(checkpoint_path, map_location=device)
        classifier.load_state_dict(checkpoint['state_dict'])
    else:
        print("Warning: Utilizing untrained weights. Valid checkpoint not found.")

    classifier.to(device)

    # 6. Aggregate predictions across all extracted patches
    mst_class, confidence = predict_patches(patches, classifier, device)

    # Format the final output string
    label = f"Monk Skin Tone (MST) {mst_class}"
    return label, confidence


# sample usage - 
# img_path = "custom-images/00048_png_jpg.rf.2cd59aace0c099f72e3fad3b12ebfc51.jpg"
# label, confidence = predict_skin_tone(img_path)
# print(f"Predicted Label: {label}")
# print(f"Confidence: {confidence:.2f}%")

# sample output - 
# Predicted Label: Monk Skin Tone (MST) 3
# Confidence: 96.33%