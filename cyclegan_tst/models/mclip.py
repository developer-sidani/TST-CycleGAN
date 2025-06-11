import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPTokenizer, CLIPProcessor
from multilingual_clip.pt_multilingual_clip import MultilingualCLIP
import transformers


class M_CLIP(nn.Module):
    """
    M-CLIP text encoder wrapper for TST-CycleGAN integration.
    Extracts text features using multilingual CLIP.
    """
    
    def __init__(self, model_name="M-CLIP/XLM-Roberta-Large-Vit-B-16Plus"):
        super(M_CLIP, self).__init__()
        
        # Initialize M-CLIP text encoder
        self.model = MultilingualCLIP.from_pretrained(model_name)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
        
        # Freeze the model parameters as they are pretrained
        for param in self.model.parameters():
            param.requires_grad = False
            
        print(f"Loaded M-CLIP model: {model_name}")
        print(f"Model frozen: {not any(p.requires_grad for p in self.model.parameters())}")
    
    def forward(self, text):
        """
        Extract text features using M-CLIP text encoder
        
        Args:
            text (List[str]): List of text sentences
            
        Returns:
            torch.Tensor: Text features with shape (batch_size, feature_dim)
        """
        # Tokenize the text
        inputs = self.tokenizer(
            text, 
            padding=True, 
            truncation=True, 
            max_length=77,  # Standard CLIP text length
            return_tensors="pt"
        )
        
        # Move to the same device as the model
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Extract features using M-CLIP
        with torch.no_grad():
            # Get transformer outputs
            transformer_outputs = self.model.transformer(**inputs)[0]
            
            # Apply attention masking and pooling
            attention_mask = inputs['attention_mask']
            masked_outputs = transformer_outputs * attention_mask.unsqueeze(2)
            pooled_outputs = masked_outputs.sum(dim=1) / attention_mask.sum(dim=1)[:, None]
            
            # Apply linear transformation
            text_features = self.model.LinearTransformation(pooled_outputs)
            
        return text_features
    
    def to(self, device):
        """Move model to device"""
        self.model = self.model.to(device)
        return self 