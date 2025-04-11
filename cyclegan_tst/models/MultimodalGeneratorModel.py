from typing import List, Optional, Tuple, Union
import os
import torch
from torch import nn
import torch.nn.functional as F

from transformers import (
    MBartForConditionalGeneration,
    MBart50TokenizerFast,
    AutoModel,
    AutoProcessor,
    AutoTokenizer
)
from tqdm import tqdm

from cyclegan_tst.models.MultimodalAdapter import MLPAdapter, HiddenMLPAdapter, TransformerAdapter
from utils.utils import get_lang_code


class MultimodalGeneratorModel(nn.Module):
    """
    Generator model for multimodal machine translation.
    Combines CLIP for image encoding, an adapter network, and mBART for translation.
    """

    def __init__(
            self,
            mbart_model_path: str = "facebook/mbart-large-50",
            clip_model_path: str = "M-CLIP/XLM-Roberta-Large-Vit-B-32",
            adapter_type: str = "mlp",
            prefix_length: int = 10,
            max_seq_length: int = 64,
            src_lang: str = "en",
            tgt_lang: str = "de",
            pretrained_path: str = None,
    ):
        super(MultimodalGeneratorModel, self).__init__()

        self.max_seq_length = max_seq_length
        self.prefix_length = prefix_length
        self.clip_model_path = clip_model_path

        # Convert short language codes to full mBART language codes
        self.src_lang = get_lang_code(src_lang)
        self.tgt_lang = get_lang_code(tgt_lang)

        # Load models from pretrained path or initialize from scratch
        if pretrained_path is None:
            # mBART for translation
            self.mbart_model = MBartForConditionalGeneration.from_pretrained(mbart_model_path)
            self.tokenizer = MBart50TokenizerFast.from_pretrained(mbart_model_path)

            # CLIP models for image/text encoding
            self.clip_model = AutoModel.from_pretrained(clip_model_path)
            self.clip_processor = AutoProcessor.from_pretrained(clip_model_path)
            self.clip_tokenizer = AutoTokenizer.from_pretrained(clip_model_path)

            # Adapter initialization
            clip_dim = 768 if "Large" in clip_model_path else 512  # Adjust based on model
            mbart_dim = self.mbart_model.config.d_model

            if adapter_type == "mlp":
                self.adapter = MLPAdapter(clip_dim=clip_dim, mbart_dim=mbart_dim, prefix_length=prefix_length)
            elif adapter_type == "hidden_mlp":
                self.adapter = HiddenMLPAdapter(clip_dim=clip_dim, mbart_dim=mbart_dim, prefix_length=prefix_length)
            elif adapter_type == "transformer":
                self.adapter = TransformerAdapter(clip_dim=clip_dim, mbart_dim=mbart_dim,
                                                  prefix_length=prefix_length, num_encoder_layers=1)
            else:
                raise ValueError(f"Unknown adapter type: {adapter_type}")
        else:
            # Load from pretrained path
            self.mbart_model = MBartForConditionalGeneration.from_pretrained(f"{pretrained_path}/mbart")
            self.tokenizer = MBart50TokenizerFast.from_pretrained(f"{pretrained_path}/tokenizer")

            self.clip_model = AutoModel.from_pretrained(f"{pretrained_path}/clip")
            self.clip_processor = AutoProcessor.from_pretrained(f"{pretrained_path}/clip_processor")
            self.clip_tokenizer = AutoTokenizer.from_pretrained(f"{pretrained_path}/clip_tokenizer")

            self.adapter = torch.load(f"{pretrained_path}/adapter.pt")

    def train(self):
        # Setting models in training mode
        self.mbart_model.train()
        # Keep CLIP frozen during training
        self.clip_model.eval()

    def eval(self):
        # Setting models in evaluation mode
        self.mbart_model.eval()
        self.clip_model.eval()

    def encode_image(self, images, device=None):
        """
        Encode images using CLIP model

        Args:
            images: List of PIL images or image paths
            device: Device to use for computation

        Returns:
            Image embeddings from CLIP
        """
        if isinstance(images[0], str):
            # Load images from paths
            from PIL import Image
            images = [Image.open(img_path).convert('RGB') for img_path in images]

        # Process images for CLIP
        inputs = self.clip_processor(images=images, return_tensors="pt")
        if device:
            inputs = {k: v.to(device) for k, v in inputs.items()}

        # Get embeddings
        with torch.no_grad():
            outputs = self.clip_model.get_image_features(**inputs)

        return outputs

    def encode_text(self, texts, device=None):
        """
        Encode text using CLIP model

        Args:
            texts: List of text strings
            device: Device to use for computation

        Returns:
            Text embeddings from CLIP
        """
        inputs = self.clip_tokenizer(texts, padding=True, truncation=True,
                                     max_length=self.max_seq_length, return_tensors="pt")
        if device:
            inputs = {k: v.to(device) for k, v in inputs.items()}

        # Get embeddings
        with torch.no_grad():
            outputs = self.clip_model.get_text_features(**inputs)

        return outputs

    def forward(
            self,
            input_data,
            is_clip_embedding=False,
            reference_sentences=None,
            device=None,
            generate_only=False
    ):
        """
        Forward pass for the multimodal generator

        Args:
            input_data: Either CLIP embeddings, texts, or images
            is_clip_embedding: Whether input_data is already CLIP embeddings
            reference_sentences: Target sentences for computing loss
            device: Device to use for computation
            generate_only: If True, only generate outputs without computing loss

        Returns:
            Depending on the mode:
            - (output_ids, decoded_sentences) for generation only
            - (output_ids, decoded_sentences, loss) when reference_sentences provided
        """
        # Get CLIP embeddings if not already provided
        if not is_clip_embedding:
            if isinstance(input_data[0], str):
                # Text input
                clip_embeddings = self.encode_text(input_data, device)
            else:
                # Image input
                clip_embeddings = self.encode_image(input_data, device)
        else:
            clip_embeddings = input_data

        # Send to device if needed
        if device and not is_clip_embedding:
            clip_embeddings = clip_embeddings.to(device)

        # Transform CLIP embeddings to mBART embeddings using adapter
        prefix_embeds = self.adapter(clip_embeddings)

        # Prepare special tokens for decoder input
        bos_token_id = self.tokenizer.lang_code_to_id[self.tgt_lang]

        # Generate translation
        if generate_only or reference_sentences is None:
            # For generation mode
            output_ids = self.mbart_model.generate(
                encoder_outputs=[prefix_embeds, None, None],  # Provide prefix embeddings as encoder output
                decoder_start_token_id=bos_token_id,
                forced_bos_token_id=bos_token_id,
                max_length=self.max_seq_length,
                num_beams=4,
            )

            # Decode the output
            decoded_sentences = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)

            return output_ids, decoded_sentences

        else:
            # For training mode with loss computation
            # Tokenize target sentences
            target_inputs = self.tokenizer(
                reference_sentences,
                padding=True,
                truncation=True,
                max_length=self.max_seq_length,
                return_tensors="pt"
            )

            if device:
                target_inputs = {k: v.to(device) for k, v in target_inputs.items()}

            # Prepare decoder inputs (shift right)
            decoder_input_ids = self.mbart_model.prepare_decoder_input_ids_from_labels(target_inputs["input_ids"])

            # Forward pass with teacher forcing
            outputs = self.mbart_model(
                encoder_outputs=[prefix_embeds, None, None],
                decoder_input_ids=decoder_input_ids,
                labels=target_inputs["input_ids"],
                return_dict=True
            )

            # Generate for returning decoded sentences
            with torch.no_grad():
                output_ids = self.mbart_model.generate(
                    encoder_outputs=[prefix_embeds, None, None],
                    decoder_start_token_id=bos_token_id,
                    forced_bos_token_id=bos_token_id,
                    max_length=self.max_seq_length,
                    num_beams=2,
                )

            # Decode the output
            decoded_sentences = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)

            return output_ids, decoded_sentences, outputs.loss

    def transfer(
            self,
            input_data,
            is_image=False,
            device=None
    ):
        """
        Simplified interface for translation

        Args:
            input_data: Source text or images
            is_image: Whether input_data contains images
            device: Device to use for computation

        Returns:
            Translated sentences
        """
        if is_image:
            # Encode images
            clip_embeddings = self.encode_image(input_data, device)
        else:
            # Encode text
            clip_embeddings = self.encode_text(input_data, device)

        # Generate translations
        _, decoded_sentences = self.forward(
            clip_embeddings,
            is_clip_embedding=True,
            device=device,
            generate_only=True
        )

        return decoded_sentences

    def save_model(
            self,
            path: str
    ):
        """Save all model components"""
        os.makedirs(path, exist_ok=True)

        # Save mBART
        self.mbart_model.save_pretrained(f"{path}/mbart")
        self.tokenizer.save_pretrained(f"{path}/tokenizer")

        # Save CLIP
        self.clip_model.save_pretrained(f"{path}/clip")
        self.clip_processor.save_pretrained(f"{path}/clip_processor")
        self.clip_tokenizer.save_pretrained(f"{path}/clip_tokenizer")

        # Save adapter
        torch.save(self.adapter, f"{path}/adapter.pt")