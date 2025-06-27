import torch
import torch.nn as nn
from transformers import AutoVideoProcessor, AutoModel

class Vjepa(nn.Module):

    def __init__(self):
        hf_repo = "facebook/vjepa2-vitl-fpc64-256"
        self.model = AutoModel.from_pretrained(hf_repo)
        self.processor = AutoVideoProcessor.from_pretrained(hf_repo)

    def forward(self,x):
        pixel_values = self.processor(x, return_tensors="pt").to(self.model.device)["pixel_values_videos"]
        pixel_values = pixel_values.repeat(1, 16, 1, 1, 1)
        with torch.no_grad():
            image_embeddings = self.model.get_vision_features(pixel_values)  
        return image_embeddings


