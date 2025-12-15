#!/usr/bin/env python
# coding: utf-8
# Notebook 06_07_inference_pipeline_recommendation converted to a python file for import into a jupyter notebook

# In[1]:


import cv2
import numpy as np
import torch
import joblib

from PIL import Image
from torchvision import models, transforms
from pathlib import Path


# In[2]:


PROJECT_ROOT = Path.cwd()
MODEL_DIR = PROJECT_ROOT / "models"


# In[3]:


device = "cpu"

model = models.resnet18(weights="IMAGENET1K_V1")
model.fc = torch.nn.Identity()
model.eval()
model.to(device)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# In[4]:


def get_embedding_from_crop(crop_bgr):
    img = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
    x = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        emb = model(x)

    return emb.squeeze().cpu().numpy()


# In[5]:


face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


# In[6]:


def detect_face(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(80, 80)
    )

    if len(faces) == 0:
        return None

    return faces[0]  # (x, y, w, h)


# In[7]:


def get_face_regions(image, face_box):
    x, y, w, h = face_box

    regions = {
        "eyes": image[
            int(y + 0.15*h):int(y + 0.45*h),
            int(x + 0.05*w):int(x + 0.95*w)
        ],

        "nose": image[
            int(y + 0.35*h):int(y + 0.65*h),
            int(x + 0.25*w):int(x + 0.75*w)
        ],

        "lips": image[
            int(y + 0.60*h):int(y + 0.90*h),
            int(x + 0.30*w):int(x + 0.70*w)
        ],

        "cheeks": image[
            int(y + 0.40*h):int(y + 0.75*h),
            int(x):int(x + w)
        ],

        "face": image[
            y:y+h,
            x:x+w
        ]
    }

    return regions


# In[8]:


REGION_ATTRIBUTE_MAP = {
    "eyes": ["Narrow_Eyes", "Arched_Eyebrows"],
    "nose": ["Big_Nose", "Pointy_Nose"],
    "face": ["Oval_Face", "Pale_Skin"],
    "cheeks": ["High_Cheekbones", "Rosy_Cheeks"]
}


# In[9]:


def load_classifiers():
    classifiers = {}

    for region, attrs in REGION_ATTRIBUTE_MAP.items():
        for attr in attrs:
            path = MODEL_DIR / f"{region}_{attr}_classifier.joblib"
            if path.exists():
                classifiers[(region, attr)] = joblib.load(path)

    return classifiers


# In[10]:


def classify_features(image_path, classifiers):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Could not load image")

    face_box = detect_face(image)
    if face_box is None:
        return {"error": "No face detected"}

    regions = get_face_regions(image, face_box)
    predictions = {}

    for (region, attr), clf in classifiers.items():
        crop = regions.get(region)
        if crop is None or crop.size == 0:
            continue

        emb = get_embedding_from_crop(crop)
        pred = clf.predict([emb])[0]

        predictions[attr] = int(pred)

    return predictions


# In[11]:


classifiers = load_classifiers()

image_path = "test_face.jpg"  # change this
predictions = classify_features(image_path, classifiers)


# In[12]:


def get_active_features(row, feature_cols):
    """
    Returns a set of features that are 'on' (value == 1) for this image.
    """
    return {feat for feat in feature_cols if row[feat] == 1}


# In[13]:


SINGLE_FEATURE_RULES = {
    "Narrow_Eyes": [
        "Use lighter eyeshadow shades on the inner corners and center of the lid to visually open the eyes.",
        "Focus eyeliner and mascara toward the outer third of the eye to create width."
    ],

    "High_Cheekbones": [
        "Apply blush slightly lower on the cheeks rather than directly on the cheekbones for balance.",
        "Use subtle contour beneath the cheekbones to enhance structure without over-definition."
    ],

    "Rosy_Cheeks": [
        "Use a green-toned color corrector before foundation to neutralize redness.",
        "Choose neutral or peach-toned blushes instead of pink to avoid emphasizing redness."
    ],

    "Pointy_Nose": [
        "Apply soft matte contour along the sides of the nose and blend thoroughly to soften sharp angles.",
        "Avoid strong highlight on the nose tip to reduce emphasis."
    ],

    "Big_Nose": [
        "Use matte contour along the sides of the nose bridge to create balance.",
        "Shift visual focus toward the eyes or lips to draw attention away from the center of the face."
    ],

    "Pale_Skin": [
        "Use soft peach or rose blush shades to add warmth to the complexion.",
        "Avoid overly dark contour shades; opt for light, neutral tones instead."
    ],

    "Oval_Face": [
        "Minimal contouring is needed; focus on blush placement to enhance natural balance.",
        "Experiment freely with eye and lip makeup, as most styles complement an oval face."
    ],

    "Arched_Eyebrows": [
        "Follow the natural brow arch and avoid flattening it with heavy filling.",
        "Use lighter brow products and softer strokes to keep the look balanced."
    ]
}
COMBO_RULES = {
    frozenset(["Narrow_Eyes", "Arched_Eyebrows"]): [
        "Keep brow makeup soft while using light-reflecting eyeshadows to open the eye area without over-sharpening features."
    ],

    frozenset(["Narrow_Eyes", "Rosy_Cheeks"]): [
        "Use brightening eye makeup to open the eyes while choosing neutral lip tones to maintain facial balance."
    ],

    frozenset(["High_Cheekbones", "Oval_Face"]): [
        "Use blush sparingly and focus on subtle highlighting to enhance structure without overpowering natural balance."
    ],

    frozenset(["Rosy_Cheeks", "Pale_Skin"]): [
        "Prioritize color correction and lightweight foundation to even skin tone while maintaining a natural finish."
    ],

    frozenset(["Pointy_Nose", "Big_Nose"]): [
        "Use soft, blended contouring techniques and avoid harsh highlights to create a more balanced nose appearance."
    ],

    frozenset(["Arched_Eyebrows", "Pointy_Nose"]): [
        "Balance strong facial angles with soft eye makeup and diffused contouring across the center of the face."
    ],

    frozenset(["High_Cheekbones", "Pale_Skin"]): [
        "Choose muted or cool-toned lip colors and balance with warm blush to avoid high contrast."
    ],

    frozenset(["High_Cheekbones", "Rosy_Cheeks"]): [
        "Apply blush lower on the cheeks and blend outward to reduce emphasis on redness while maintaining structure."
    ]
}


# In[14]:


def generate_recommendations(active_features):
    recs = []

    # 1. Check combo rules
    for combo, combo_recs in COMBO_RULES.items():
        if combo.issubset(active_features):
            recs.extend(combo_recs)

    # 2. Single-feature rules
    for feat in active_features:
        if feat in SINGLE_FEATURE_RULES:
            recs.extend(SINGLE_FEATURE_RULES[feat])

    # 3. Remove duplicates while preserving order
    seen = set()
    final_recs = []
    for r in recs:
        if r not in seen:
            final_recs.append(r)
            seen.add(r)

    return final_recs


# In[15]:


# Step 6 output (already computed earlier)
# predictions = classify_features(image_path, classifiers)

# Convert Step 6 predictions into active feature set
active_features = {feat for feat, val in predictions.items() if val == 1}

# Step 7: generate recommendations
recommendations = generate_recommendations(active_features)

