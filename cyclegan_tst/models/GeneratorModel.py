from typing import List, Optional, Tuple, Union
from torch import nn
import torch

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, MBartForConditionalGeneration, MBart50TokenizerFast
from transformers import CLIPModel, AutoProcessor, CLIPProcessor, CLIPVisionModel
from tqdm import tqdm
from utils.utils import get_lang_code

class MLPAdapter(nn.Module):
    def __init__(self, clip_dim=512, mbart_dim=1024, prefix_length=10):
        super(MLPAdapter, self).__init__()
        self.hidden = nn.Sequential(
            nn.Linear(clip_dim, prefix_length * mbart_dim),
            nn.PReLU(),
        )
        self.prefix_length = prefix_length
        self.mbart_dim = mbart_dim
    
    def forward(self, x):
        x = x.float()
        return self.hidden(x).view(-1, self.prefix_length, self.mbart_dim)
    
    def reset(self):
        nn.init.xavier_uniform_(self.hidden[0].weight)

class HiddenMLPAdapter(nn.Module):
    def __init__(self, clip_dim=512, mbart_dim=1024, prefix_length=10):
        super(HiddenMLPAdapter, self).__init__()
        self.hidden = nn.Sequential(
            nn.Linear(clip_dim, mbart_dim),
            nn.ReLU(),
            nn.Linear(mbart_dim, prefix_length * mbart_dim),
            nn.PReLU(),
        )
        self.prefix_length = prefix_length
        self.mbart_dim = mbart_dim
    
    def forward(self, x):
        x = x.float()
        return self.hidden(x).view(-1, self.prefix_length, self.mbart_dim)
    
    def reset(self):
        nn.init.xavier_uniform_(self.hidden[0].weight)
        nn.init.xavier_uniform_(self.hidden[2].weight)

class TransformerAdapter(nn.Module):
    def __init__(self, clip_dim=512, mbart_dim=1024, prefix_length=10, num_encoder_layers=1):
        super(TransformerAdapter, self).__init__()
        self.prefix_length = prefix_length
        self.mbart_dim = mbart_dim
        self.projector = nn.Linear(clip_dim, prefix_length * mbart_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=mbart_dim,
            nhead=2,
            dim_feedforward=mbart_dim//3,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
                encoder_layer=encoder_layer, 
                num_layers=num_encoder_layers
            )
    
    def forward(self, x):
        x = self.projector(x).view(-1, self.prefix_length, self.mbart_dim)
        return self.transformer(x)

class GeneratorModel(nn.Module):
    
    def __init__(
        self,
        model_name_or_path: str,
        pretrained_path: str = None,
        max_seq_length: int = 64,
        truncation: str = "longest_first",
        padding: str = "max_length",
        src_lang: str = "en",  # Short language code
        tgt_lang: str = "de",   # Short language code
        use_clip: bool = False,
        clip_model_name: str = "openai/clip-vit-base-patch32",
        prefix_length: int = 10,
        mapping_network: str = "mlp"
        ):
        super(GeneratorModel, self).__init__()
        
        self.model_name_or_path = model_name_or_path
        self.max_seq_length = max_seq_length
        self.truncation = truncation
        self.padding = padding
        self.use_clip = use_clip
        self.prefix_length = prefix_length
        
        # Convert short language codes to full mBART language codes
        self.src_lang = get_lang_code(src_lang)
        self.tgt_lang = get_lang_code(tgt_lang)
        
        if pretrained_path is None:
            self.model = MBartForConditionalGeneration.from_pretrained(model_name_or_path)
            self.tokenizer = MBart50TokenizerFast.from_pretrained(model_name_or_path)
        else:
            self.model = MBartForConditionalGeneration.from_pretrained(pretrained_path)
            self.tokenizer = MBart50TokenizerFast.from_pretrained(f"{pretrained_path}tokenizer/")
        
        # CLIP and adapter integration
        if self.use_clip:
            self.clip_model_name = clip_model_name
            self.clip = CLIPVisionModel.from_pretrained(clip_model_name)
            self.clip_processor = AutoProcessor.from_pretrained(clip_model_name)
            # Freeze CLIP model
            for param in self.clip.parameters():
                param.requires_grad = False
            
            # Initialize the adapter based on the chosen mapping network
            clip_dim = 512  # Standard CLIP dimension for most models
            mbart_dim = 1024  # Standard mBART dimension
            
            if mapping_network == "mlp":
                self.adapter = MLPAdapter(clip_dim=clip_dim, mbart_dim=mbart_dim, prefix_length=prefix_length)
            elif mapping_network == "hidden_mlp":
                self.adapter = HiddenMLPAdapter(clip_dim=clip_dim, mbart_dim=mbart_dim, prefix_length=prefix_length)
            elif mapping_network == "transformer":
                self.adapter = TransformerAdapter(clip_dim=clip_dim, mbart_dim=mbart_dim, prefix_length=prefix_length, num_encoder_layers=1)
            else:
                raise ValueError(f"Unsupported mapping network: {mapping_network}")

    def train(self):
        # Setting the model in training mode
        self.model.train()

    def eval(self):
        # Setting the model in evaluation mode
        self.model.eval()
    
    def get_clip_embeddings(self, images, device=None):
        """Extract CLIP embeddings from images."""
        processed_images = self.clip_processor(images=images, return_tensors="pt")
        if device:
            processed_images = {k: v.to(device) for k, v in processed_images.items()}
            
        with torch.no_grad():
            image_features = self.clip(**processed_images).pooler_output
        
        return image_features
    
    def forward_with_clip(
        self,
        images,
        target_sentences: List[str] = None,
        device = None,
        ):
        """Forward pass using CLIP embeddings as input."""
        # Debug: Print image information
        print(f"DEBUG: Processing {len(images)} images")
        if isinstance(images[0], torch.Tensor):
            print(f"DEBUG: Image tensor shape: {images[0].shape}")
        
        # Get CLIP embeddings
        clip_embeddings = self.get_clip_embeddings(images, device)
        print(f"DEBUG: CLIP embeddings shape: {clip_embeddings.shape}")
        
        # Process through adapter
        prefix_embeds = self.adapter(clip_embeddings)
        print(f"DEBUG: Prefix embeddings shape: {prefix_embeds.shape}")
        
        if target_sentences is not None:
            print(f"DEBUG: Target sentences: {target_sentences[0][:50]}...")
            
            # Get tokenized target sentences
            labels = self.tokenizer(target_sentences,
                truncation=self.truncation, 
                padding=self.padding, 
                max_length=self.max_seq_length,
                return_tensors="pt").input_ids.to(device)
            
            print(f"DEBUG: Target labels shape: {labels.shape}")
            
            # From the labels, create decoder_input_ids
            from transformers.models.mbart.modeling_mbart import shift_tokens_right
            decoder_input_ids = shift_tokens_right(labels, self.tokenizer.pad_token_id)
            
            # Get decoder embeddings
            decoder_inputs_embeds = self.model.model.decoder.embed_tokens(decoder_input_ids) * self.model.model.decoder.embed_scale
            
            # Concatenate prefix embeddings with decoder embeddings
            decoder_inputs_embeds = torch.cat((prefix_embeds, decoder_inputs_embeds[:, 1:]), dim=1)
            
            # Create decoder attention mask
            decoder_attention_mask = torch.ones_like(decoder_input_ids).to(device)
            decoder_attention_mask[decoder_input_ids == self.tokenizer.pad_token_id] = 0
            
            # Extend decoder attention mask for prefix embeddings
            decoder_attention_mask = torch.cat((
                torch.ones(prefix_embeds.shape[0], prefix_embeds.shape[1]).to(device),
                decoder_attention_mask[:, 1:]
            ), dim=1)
            
            # Modify labels to accommodate prefix embeddings
            dummy_tokens = (torch.ones((labels.shape[0], self.prefix_length)) * self.tokenizer.pad_token_id).to(device)
            modified_labels = torch.cat((dummy_tokens, labels[:, 1:]), dim=1).long()
            
            # Set padding tokens to -100 to ignore in loss calculation
            modified_labels[modified_labels == self.tokenizer.pad_token_id] = -100
            
            # Forward pass through the model
            outputs = self.model(
                decoder_inputs_embeds=decoder_inputs_embeds,
                decoder_attention_mask=decoder_attention_mask,
                labels=modified_labels
            )
            
            print(f"DEBUG: Loss from model: {outputs.loss}")
            
            # Generate text for return
            generated_ids = self.model.generate(
                decoder_inputs_embeds=prefix_embeds,
                decoder_attention_mask=torch.ones((prefix_embeds.shape[0], prefix_embeds.shape[1])).to(device),
                forced_bos_token_id=self.tokenizer.lang_code_to_id[self.tgt_lang],
                max_length=self.max_seq_length
            )
            
            transferred_sentences = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            print(f"DEBUG: Generated {len(transferred_sentences)} sentences. First example: {transferred_sentences[0][:50]}...")
            return generated_ids, transferred_sentences, outputs.loss
        
        else:
            # Generate text directly from prefix embeddings
            generated_ids = self.model.generate(
                decoder_inputs_embeds=prefix_embeds,
                decoder_attention_mask=torch.ones((prefix_embeds.shape[0], prefix_embeds.shape[1])).to(device),
                forced_bos_token_id=self.tokenizer.lang_code_to_id[self.tgt_lang],
                max_length=self.max_seq_length
            )
            
            transferred_sentences = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            print(f"DEBUG: Generated {len(transferred_sentences)} sentences. First example: {transferred_sentences[0][:50]}...")
            return generated_ids, transferred_sentences
    
    def forward(
        self,
        sentences: List[str] = None,
        target_sentences: List[str] = None,
        images = None,
        device = None,
        ):
        """Forward pass that can handle either text or images as input."""
        # If CLIP is enabled and images are provided, use the CLIP pathway
        if self.use_clip and images is not None:
            print("DEBUG: Using CLIP pathway for forward pass")
            return self.forward_with_clip(images, target_sentences, device)
        
        # Otherwise, use the standard text-to-text pathway
        if sentences is None:
            raise ValueError("Either sentences or images must be provided")
            
        print(f"DEBUG: Processing {len(sentences)} text inputs")
        print(f"DEBUG: First sentence: {sentences[0][:50]}...")
            
        # Tokenize input sentences
        inputs = self.tokenizer(sentences, 
            truncation=self.truncation, 
            padding=self.padding, 
            max_length=self.max_seq_length,
            return_tensors="pt")
            
        print(f"DEBUG: Input tokens shape: {inputs.input_ids.shape}")

        if target_sentences is not None:
            print(f"DEBUG: Target sentence: {target_sentences[0][:50]}...")
            labels = self.tokenizer(target_sentences,
                truncation=self.truncation, 
                padding=self.padding, 
                max_length=self.max_seq_length,
                return_tensors="pt").input_ids
                
            print(f"DEBUG: Target tokens shape: {labels.shape}")
            
            inputs = inputs.to(device)
            labels = labels.to(device)
            output_supervised = self.model(**inputs, labels=labels)
            print(f"DEBUG: Supervised loss: {output_supervised.loss}")
        
        inputs = inputs.to(device)
        output = self.model.generate(
            **inputs, 
            forced_bos_token_id=self.tokenizer.lang_code_to_id[self.tgt_lang],
            max_length=self.max_seq_length
        )
        
        print(f"DEBUG: Generated tokens shape: {output.shape}")

        # Decode the output
        transferred_sentences = self.tokenizer.batch_decode(output, skip_special_tokens=True)
        print(f"DEBUG: First transferred sentence: {transferred_sentences[0][:50]}...")

        if target_sentences is not None:
            return output, transferred_sentences, output_supervised.loss
        else:
            return output, transferred_sentences
        
    def transfer(
        self,
        sentences: List[str] = None,
        images = None,
        device = None
        ):
        """Interface for generating outputs from either text or images."""
        if self.use_clip and images is not None:
            _, transferred_sentences = self.forward_with_clip(images, None, device)
            return transferred_sentences
            
        if sentences is None:
            raise ValueError("Either sentences or images must be provided")
            
        inputs = self.tokenizer(sentences, 
            truncation=self.truncation, 
            padding=self.padding, 
            max_length=self.max_seq_length,
            return_tensors="pt")
                
        inputs = inputs.to(device)
        output = self.model.generate(
            **inputs, 
            forced_bos_token_id=self.tokenizer.lang_code_to_id[self.tgt_lang],
            max_length=self.max_seq_length
        )
        
        # Decode
        transferred_sentences = self.tokenizer.batch_decode(output, skip_special_tokens=True)

        return transferred_sentences
    
    def save_model(
        self, 
        path: Union[str]
        ):
        """Save the model, tokenizer and adapter if present."""
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(f"{path}tokenizer/")
        
        # Save adapter if it exists
        if self.use_clip:
            adapter_path = f"{path}adapter.pt"
            torch.save(self.adapter.state_dict(), adapter_path)
