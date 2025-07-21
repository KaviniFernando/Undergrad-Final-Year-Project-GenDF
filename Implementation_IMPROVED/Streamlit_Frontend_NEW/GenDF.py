import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import efficientnet_b0
from torchvision.models.efficientnet import EfficientNet  # Required for safe unpickling
from PIL import Image
from torch.nn import Sequential
import torch.serialization


#For GRAD-CAM utility
import numpy as np
import matplotlib.pyplot as plt
import cv2
from torchvision.transforms.functional import to_pil_image
from torch.nn import functional as F



# Set page config
st.set_page_config(
    page_title="GenDF",
    page_icon="images/favicon.png"
)

# -------------------------------
# TTT Adapter (for feature adaptation)
# -------------------------------
class TTTAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(128, 128)

    def forward(self, x):
        return self.fc(x)


# -------------------------------
# Version C model: EfficientNet + Adapter + Classifier
# -------------------------------
 
class RL_CL_TTT_EfficientNet(nn.Module):
    def __init__(self, backbone, adapter):
        super().__init__()
        self.backbone = backbone

        # Add projector inside the model
        self.projector = nn.Sequential(
            nn.Linear(1280, 512),
            nn.ReLU(),
            nn.Linear(512, 128)
        )

        self.adapter = adapter
        self.classifier = nn.Linear(128, 2)

    def forward(self, x):
        x = self.backbone.features(x)
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)       # (B, 1280)
        x = self.projector(x)         # (B, 128)
        x = self.adapter(x)           # Still (B, 128)
        return self.classifier(x)


    
# For RL_EfficientNet (Version A)
class EfficientNetClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = efficientnet_b0(weights=None)
        self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, 2)

    def forward(self, x):
        return self.model(x)
    

class RL_EfficientNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = efficientnet_b0(weights=None)
        self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, 2)

    def forward(self, x):
        return self.model(x)


# For RL_CL_EfficientNet (Version B)
class ContrastiveEfficientNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = efficientnet_b0(weights=None)
        self.base.classifier[1] = nn.Identity()
        self.projector = nn.Sequential(
            nn.Linear(1280, 512),
            nn.ReLU(),
            nn.Linear(512, 128)
        )
        self.classifier = nn.Linear(128, 2)

    def forward(self, x):
        features = self.base(x)
        z = self.projector(features)
        return self.classifier(z)
    
class RL_CL_EfficientNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = efficientnet_b0(weights=None)
        self.base.classifier[1] = nn.Identity()
        self.projector = nn.Sequential(
            nn.Linear(1280, 512),
            nn.ReLU(),
            nn.Linear(512, 128)
        )
        self.classifier = nn.Linear(128, 2)

    def forward(self, x):
        features = self.base(x)
        z = self.projector(features)
        return self.classifier(z)


# -------------------------------
# Safe loading with full model object
# -------------------------------
torch.serialization.add_safe_globals([
    RL_CL_TTT_EfficientNet,
    TTTAdapter,
    ContrastiveEfficientNet,
    EfficientNetClassifier,
    RL_EfficientNet,
    RL_CL_EfficientNet,
    Sequential
])



# Load model function
@st.cache_resource()
def load_model(model_choice):
    if model_choice == "Version A (RL only)":
        model = torch.load("gendf_versionA.pth", map_location=torch.device("cpu"), weights_only=False)
    elif model_choice == "Version B (RL+CL)":
        model = torch.load("gendf_versionB.pth", map_location=torch.device("cpu"), weights_only=False)
    elif model_choice == "Version C (RL+CL+TTT)":
        model = torch.load("gendf_versionC.pth", map_location=torch.device("cpu"), weights_only=False)
    else:  # Version D (CL only)
        model = torch.load("gendf_versionD.pth", map_location=torch.device("cpu"), weights_only=False)

    model.eval()
    return model


# -------------------------------
# Inference with frozen TTT adapter
# -------------------------------
    
def test_time_training(model, image_tensor, model_choice):
    if model_choice == "Version C (RL+CL+TTT)":
        model.backbone.eval()
        model.adapter.eval()
        model.projector.eval()  # Include projector

        with torch.no_grad():
            features = model.backbone.features(image_tensor)
            features = model.backbone.avgpool(features)
            features = torch.flatten(features, 1)  # Shape: (1, 1280)

            features = model.projector(features)   
            adapted = model.adapter(features)
            logits = model.classifier(adapted)
        return logits
    else:
        with torch.no_grad():
            logits = model(image_tensor)
        return logits




# -------------------------------
# Image preprocessing and prediction
# -------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def preprocess_image(image):
    return transform(image).unsqueeze(0)



def predict(image):
    image_tensor = preprocess_image(image)

    # Dynamically select target layer for Grad-CAM
    if model_choice == "Version A (RL only)":
        # If loaded as full EfficientNetClassifier model
        if hasattr(model, "model"):
            target_layer = model.model.features[-1]
        else:
            target_layer = model.features[-1]
    elif model_choice == "Version B (RL+CL)":
        target_layer = model.base.features[-1]
    elif model_choice == "Version C (RL+CL+TTT)":
        target_layer = model.backbone.features[-1]
    else:  # Version D (CL only)
        target_layer = model.base.features[-1]

    cam = GradCAM(model, target_layer)
    logits = test_time_training(model, image_tensor, model_choice)
    probs = F.softmax(logits, dim=1)[0]
    class_idx = torch.argmax(probs).item()

    heatmap = cam.generate(image_tensor, class_idx, model_choice)
    cam.remove_hooks()

    return class_idx, probs, heatmap





# -------------------------------
# Explainability Module
# -------------------------------

#GRAD-CAM code
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_handles = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.hook_handles.append(self.target_layer.register_forward_hook(forward_hook))
        self.hook_handles.append(self.target_layer.register_backward_hook(backward_hook))

    def remove_hooks(self):
        for handle in self.hook_handles:
            handle.remove()

    def generate(self, input_tensor, class_idx=None, model_choice=None):
        self.model.zero_grad()
        
        if model_choice == "Version C (RL+CL+TTT)":
            features = self.model.backbone.features(input_tensor)
            x = self.model.backbone.avgpool(features)
            x = torch.flatten(x, 1)
            x = self.model.projector(x)
            x = self.model.adapter(x)
            logits = self.model.classifier(x)
            
        elif model_choice == "Version B (RL+CL)":
            features = self.model.base.features(input_tensor)
            x = self.model.base.avgpool(features)
            x = torch.flatten(x, 1)
            x = self.model.projector(x)
            logits = self.model.classifier(x)
            
        elif model_choice == "Version A (RL only)":
            if hasattr(self.model, "model"):
                logits = self.model(input_tensor)
                features = self.model.model.features(input_tensor)
            else:
                logits = self.model(input_tensor)
                features = self.model.features(input_tensor)
                
        elif model_choice == "Version D (CL only)":
            features = self.model.base.features(input_tensor)
            x = self.model.base.avgpool(features)
            x = torch.flatten(x, 1)
            x = self.model.projector(x)
            logits = self.model.classifier(x)
        else:
            return None

        
        if class_idx is None:
            class_idx = torch.argmax(logits)
        
        class_score = logits[0, class_idx]
        class_score.backward()
            
        if self.gradients is None or self.activations is None:
            return None
        
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        cam = cv2.resize(cam, (224, 224))

        return cam


#simple explaination
def explain_prediction(class_idx, confidence):
    if confidence > 0.9:
        return "Very high confidence — features strongly match known patterns."
    elif confidence > 0.7:
        return "Confident prediction based on key visual patterns in facial structure."
    elif confidence > 0.55:
        return "Moderate confidence — could be borderline. Features are ambiguous."
    else:
        return "Low confidence — image has characteristics that overlap both real and fake."

# -------------------------------
# Streamlit UI
# -------------------------------
st.image("images/icon.png", width=180)

st.title("GenDF: Deepfake Image Detector")

model_choice = st.radio(
    "**Select prefered Model Version**",
    ["Version A (RL only)", "Version B (RL+CL)", "Version C (RL+CL+TTT)", "Version D (CL only)"],
    horizontal=True
)



model = load_model(model_choice)

# Freeze backbone and adapter for inference
if model_choice == "Version C (RL+CL+TTT)":
    for param in model.backbone.parameters():
        param.requires_grad = False
    for param in model.adapter.parameters():
        param.requires_grad = False


st.write(f"**Using model:** `{model_choice}`")


uploaded_file = st.file_uploader("*Upload an image*", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image Preview", width=300)

    st.subheader("Prediction")
   
    class_idx, probs, heatmap = predict(image)
    labels = ["Real", "Fake"]
    confidence = probs[class_idx].item()

    st.markdown(f"**This image is** `{labels[class_idx]}`")
    st.markdown(f"**Confidence:** `{confidence:.4f}`")
    st.progress(confidence)

    # Show explanation
    st.markdown("**Explanation on confidence rate:**")
    st.info(explain_prediction(class_idx, confidence))


    
    # Show Grad-CAM overlay
    # # Convert heatmap to overlay on original image
    image_np = np.array(image.resize((224, 224)))
    
    if heatmap is not None:
        heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(image_np, 0.6, heatmap_color, 0.4, 0)
        
        if st.checkbox("🔍 Show Grad-CAM explanation"):
            st.image(overlay, caption="Grad-CAM: Important Regions", use_column_width=True)
        else:
            st.warning("select the checkbox if you want a Grad_CAM preview. (This highlights the image regions that the model found most important — red means high attention, blue means low.)")



# -------------------------------
# Footer
# -------------------------------
st.markdown("""
    <style>
    .footer {
        position: fixed;
        bottom: 0;
        width: 100%;
        padding: 10px;
        color: grey;
        background-color: white;
        text-align: center;
        font-size: 14px;
        border-top: 1px solid #f0f0f0;
    }
    </style>
    <div class="footer">
        © 2025 Kavini Fernando | GenDF Final Year Project
    </div>
""", unsafe_allow_html=True)





