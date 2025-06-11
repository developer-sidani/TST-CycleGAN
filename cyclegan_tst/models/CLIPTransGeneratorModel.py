from typing import List, Optional, Tuple, Union
import torch
from torch import nn
import os

from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
from transformers.models.mbart.modeling_mbart import shift_tokens_right
from utils.utils import get_lang_code
from .mclip import M_CLIP
from .adapters import MLPAdapter, HiddenMLPAdapter, TransformerAdapter

class CLIPTransGeneratorModel(nn.Module):
    """
    CLIPTrans-based generator model for TST-CycleGAN.
    Integrates M-CLIP text encoder, mapping network, and mBART.
    """
    
    def __init__(
        self,
        model_name_or_path: str,
        pretrained_path: str = None,
        max_seq_length: int = 64,
        truncation: str = "longest_first",
        padding: str = "max_length",
        src_lang: str = "en",  # Short language code
        tgt_lang: str = "de",   # Short language code
        prefix_length: int = 10,
        mapping_network: str = "mlp",
        tokenizer_path: str = None,
    ):
        super(CLIPTransGeneratorModel, self).__init__()
        
        print(f"[DEBUG] CLIPTransGeneratorModel init:")
        print(f"  src_lang: {src_lang}")
        print(f"  tgt_lang: {tgt_lang}")
        print(f"  pretrained_path: {pretrained_path}")
        print(f"  prefix_length: {prefix_length}")
        print(f"  mapping_network: {mapping_network}")
        
        self.prefix_length = prefix_length
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.max_seq_length = max_seq_length
        self.truncation = truncation
        self.padding = padding
        
        # Initialize M-CLIP text encoder
        self.mclip = M_CLIP()
        
        # Detect actual M-CLIP output dimension
        print("[DEBUG] Detecting M-CLIP output dimension...")
        test_text = ["test sentence"]
        with torch.no_grad():
            test_output = self.mclip(test_text)
            detected_clip_dim = test_output.shape[-1]
        print(f"[DEBUG] Detected M-CLIP output dimension: {detected_clip_dim}")
        
        # Check if we have a checkpoint to determine the expected dimension
        expected_clip_dim = detected_clip_dim
        checkpoint_to_check = pretrained_path
        
        # If no pretrained_path, check if we're in a directory with CycleGAN checkpoint files
        if not checkpoint_to_check and model_name_or_path:
            # Check if model_name_or_path contains adapter.pt (indicating it's a checkpoint directory)
            adapter_file = os.path.join(model_name_or_path, "adapter.pt")
            if os.path.exists(adapter_file):
                checkpoint_to_check = model_name_or_path
                print(f"[DEBUG] Found CycleGAN checkpoint directory for dimension check: {checkpoint_to_check}")
        
        if checkpoint_to_check and os.path.exists(checkpoint_to_check):
            # Try to load checkpoint to check adapter dimensions
            possible_files = [
                os.path.join(checkpoint_to_check, "adapter.pt"),  # CycleGAN checkpoint
                os.path.join(checkpoint_to_check, "model_pretrained.pth"),  # Stage 1 checkpoint
                os.path.join(checkpoint_to_check, "model_best_test.pth"),  # Stage 1 checkpoint
            ]
            for checkpoint_file in possible_files:
                if os.path.exists(checkpoint_file):
                    try:
                        if checkpoint_file.endswith("adapter.pt"):
                            # CycleGAN adapter checkpoint
                            adapter_state = torch.load(checkpoint_file, map_location='cpu')
                            if 'hidden.0.weight' in adapter_state:
                                adapter_weight_shape = adapter_state['hidden.0.weight'].shape
                                expected_clip_dim = adapter_weight_shape[1]  # Input dimension
                                print(f"[DEBUG] Found CycleGAN adapter expecting CLIP dim: {expected_clip_dim}")
                                break
                        else:
                            # Stage 1 checkpoint
                            checkpoint = torch.load(checkpoint_file, map_location='cpu')
                            if 'model' in checkpoint:
                                model_state = checkpoint['model']
                                if 'adapter.hidden.0.weight' in model_state:
                                    # Extract expected dimension from adapter weights
                                    adapter_weight_shape = model_state['adapter.hidden.0.weight'].shape
                                    expected_clip_dim = adapter_weight_shape[1]  # Input dimension
                                    print(f"[DEBUG] Found Stage 1 adapter expecting CLIP dim: {expected_clip_dim}")
                                    break
                    except Exception as e:
                        print(f"[DEBUG] Could not check checkpoint dimensions from {checkpoint_file}: {e}")
                        continue
        
        # Handle dimension mismatch with projection layer
        self.clip_projection = None
        if detected_clip_dim != expected_clip_dim:
            print(f"[DEBUG] M-CLIP dimension mismatch: detected {detected_clip_dim}, expected {expected_clip_dim}")
            print(f"[DEBUG] Adding projection layer: {detected_clip_dim} -> {expected_clip_dim}")
            self.clip_projection = nn.Linear(detected_clip_dim, expected_clip_dim)
        
        # Use the expected dimension for adapter
        actual_clip_dim = expected_clip_dim
        print(f"[DEBUG] Using M-CLIP dimension for adapter: {actual_clip_dim}")
        
        # Initialize adapter with correct dimensions
        mbart_dim = 1024  # mBART embedding dimension
        adapter_inputs = {
            'clip_dim': actual_clip_dim,  # Use expected dimension
            'mbart_dim': mbart_dim, 
            'prefix_length': self.prefix_length
        }
        
        if mapping_network == 'mlp':
            self.adapter = MLPAdapter(**adapter_inputs)
        elif mapping_network == 'transformer':
            self.adapter = TransformerAdapter(num_encoder_layers=1, **adapter_inputs)
        else:
            self.adapter = HiddenMLPAdapter(**adapter_inputs)
        
        print(f"[DEBUG] Adapter created with inputs: {adapter_inputs}")
        
        # Initialize mBART tokenizer and model
        # Handle tokenizer loading with flexible path resolution
        if tokenizer_path is not None:
            # Explicit tokenizer path provided
            self.tokenizer = MBart50TokenizerFast.from_pretrained(tokenizer_path)
        else:
            # Check if model_name_or_path is a checkpoint directory
            tokenizer_subdir = os.path.join(model_name_or_path, "tokenizer")
            if os.path.isdir(tokenizer_subdir):
                self.tokenizer = MBart50TokenizerFast.from_pretrained(tokenizer_subdir)
            elif os.path.exists(os.path.join(model_name_or_path, "pytorch_model.bin")):
                # This is a checkpoint directory, but no tokenizer subdirectory
                # Fall back to original mBART tokenizer
                print(f"[DEBUG] Checkpoint directory detected, using facebook/mbart-large-50 tokenizer")
                self.tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50")
            else:
                # Fall back to model_name_or_path (for original model names like facebook/mbart-large-50-many-to-many-mmt)
                self.tokenizer = MBart50TokenizerFast.from_pretrained(model_name_or_path)
        
        # Initialize mBART model
        if os.path.exists(os.path.join(model_name_or_path, "pytorch_model.bin")):
            # Load from checkpoint directory
            print(f"[DEBUG] Loading mBART from checkpoint directory: {model_name_or_path}")
            self.mbart = MBartForConditionalGeneration.from_pretrained(model_name_or_path)
        else:
            # Load from HuggingFace model name
            print(f"[DEBUG] Loading mBART from HuggingFace: {model_name_or_path}")
            self.mbart = MBartForConditionalGeneration.from_pretrained(model_name_or_path)
        
        # Load weights based on path type
        if pretrained_path and os.path.exists(pretrained_path):
            # Loading from CLIPTrans Stage 1 checkpoint
            self._load_cliptrans_weights(pretrained_path)
        elif os.path.exists(os.path.join(model_name_or_path, "adapter.pt")):
            # Loading from CycleGAN checkpoint (adapter.pt + pytorch_model.bin)
            print(f"[DEBUG] Auto-loading from CycleGAN checkpoint: {model_name_or_path}")
            self._load_cyclegan_checkpoint(model_name_or_path)
        else:
            print(f"[DEBUG] No CLIPTrans weights loaded - checkpoint path: {pretrained_path}")
        
        # Set target language token
        self.target_lang_token = get_lang_code(self.tgt_lang)
        
        print(f"[DEBUG] CLIPTransGeneratorModel initialized successfully!")

    @property
    def model(self):
        """
        Compatibility property for CycleGANModel.
        Returns the mBART model which is the main trainable component.
        """
        return self.mbart

    def parameters(self, recurse: bool = True):
        """
        Return parameters for optimization.
        Includes both mBART and adapter parameters, but excludes M-CLIP (frozen).
        """
        for param in self.mbart.parameters(recurse):
            yield param
        for param in self.adapter.parameters(recurse):
            yield param
        # Include projection layer parameters if it exists
        if self.clip_projection is not None:
            for param in self.clip_projection.parameters(recurse):
                yield param

    def _load_cliptrans_weights(self, checkpoint_path):
        """Load adapter and mBART weights from CLIPTrans Stage 1 checkpoint (same as CLIPTrans Stage 2 loading)"""
        try:
            print(f"[DEBUG] Loading CLIPTrans Stage 1 weights from: {checkpoint_path}")
            
            # Load Stage 1 checkpoint (model_pretrained.pth) - same as CLIPTrans Stage 2 does
            # Stage 1 has trained adapter AND fine-tuned mBART for captioning
            checkpoint_file = os.path.join(checkpoint_path, "model_pretrained.pth")
            
            if not os.path.exists(checkpoint_file):
                print(f"[DEBUG] No Stage 1 checkpoint file found: {checkpoint_file}")
                return
            
            print(f"[DEBUG] Loading Stage 1 checkpoint from: {checkpoint_file}")
            checkpoint = torch.load(checkpoint_file, map_location='cpu')
            
            # CLIPTrans Stage 1 checkpoint structure: {'model': full_state_dict, 'optimizer': ..., 'epoch': ...}
            if 'model' not in checkpoint:
                print(f"[DEBUG] Invalid checkpoint format - missing 'model' key")
                return
            
            model_state = checkpoint['model']
            print(f"[DEBUG] Found {len(model_state)} parameters in checkpoint")
            
            # Extract adapter and mBART weights from Stage 1 checkpoint (same as CLIPTrans Stage 2)
            adapter_state_dict = {}
            mbart_state_dict = {}
            
            for key, value in model_state.items():
                if key.startswith('adapter.'):
                    # Remove 'adapter.' prefix for our adapter
                    new_key = key.replace('adapter.', '')
                    adapter_state_dict[new_key] = value
                elif key.startswith('mbart.'):
                    # Remove 'mbart.' prefix for our mBART model
                    new_key = key.replace('mbart.', '')
                    mbart_state_dict[new_key] = value
                # Skip clip.* weights - M-CLIP is frozen and loaded separately
            
            print(f"[DEBUG] Found {len(adapter_state_dict)} adapter parameters")
            print(f"[DEBUG] Found {len(mbart_state_dict)} mBART parameters")
            
            # Load adapter weights
            if adapter_state_dict:
                try:
                    missing_keys, unexpected_keys = self.adapter.load_state_dict(adapter_state_dict, strict=False)
                    print(f"[DEBUG] Successfully loaded adapter weights")
                    print(f"[DEBUG] Adapter missing keys: {len(missing_keys)}")
                    print(f"[DEBUG] Adapter unexpected keys: {len(unexpected_keys)}")
                    if missing_keys:
                        print(f"[DEBUG] Missing adapter keys: {missing_keys[:5]}...")  # Show first 5
                except Exception as e:
                    print(f"[DEBUG] Error loading adapter weights: {e}")
                    print(f"[DEBUG] Available adapter keys: {list(adapter_state_dict.keys())[:5]}...")
                    print(f"[DEBUG] Expected adapter keys: {list(self.adapter.state_dict().keys())}")
            else:
                print(f"[DEBUG] No adapter weights found in checkpoint")
            
            # Load mBART weights (Stage 1 fine-tuned for captioning)
            if mbart_state_dict:
                try:
                    missing_keys, unexpected_keys = self.mbart.load_state_dict(mbart_state_dict, strict=False)
                    print(f"[DEBUG] Successfully loaded mBART weights from Stage 1")
                    print(f"[DEBUG] mBART missing keys: {len(missing_keys)}")
                    print(f"[DEBUG] mBART unexpected keys: {len(unexpected_keys)}")
                    if missing_keys:
                        print(f"[DEBUG] Missing mBART keys: {missing_keys[:5]}...")  # Show first 5
                except Exception as e:
                    print(f"[DEBUG] Error loading mBART weights: {e}")
            else:
                print(f"[DEBUG] No mBART weights found in checkpoint - using pre-trained weights")
            
        except Exception as e:
            print(f"[DEBUG] Error loading CLIPTrans Stage 1 weights: {e}")

    def _load_cyclegan_checkpoint(self, checkpoint_path):
        """Load adapter and mBART weights from CycleGAN checkpoint"""
        try:
            print(f"[DEBUG] Loading CLIPTrans weights from CycleGAN checkpoint: {checkpoint_path}")
            
            # Load adapter weights from adapter.pt and check dimensions
            adapter_file = os.path.join(checkpoint_path, "adapter.pt")
            if os.path.exists(adapter_file):
                print(f"[DEBUG] Loading adapter weights from: {adapter_file}")
                adapter_state_dict = torch.load(adapter_file, map_location='cpu')
                
                # Check if adapter expects different input dimension
                if 'hidden.0.weight' in adapter_state_dict:
                    saved_adapter_input_dim = adapter_state_dict['hidden.0.weight'].shape[1]
                    with torch.no_grad():
                        current_clip_dim = self.mclip(["test"]).shape[-1]
                    
                    print(f"[DEBUG] Saved adapter expects input dim: {saved_adapter_input_dim}")
                    print(f"[DEBUG] Current M-CLIP output dim: {current_clip_dim}")
                    
                    if saved_adapter_input_dim != current_clip_dim:
                        print(f"[DEBUG] Dimension mismatch detected! Need projection layer: {current_clip_dim} -> {saved_adapter_input_dim}")
                        
                        # Create projection layer if it doesn't exist
                        if self.clip_projection is None:
                            self.clip_projection = nn.Linear(current_clip_dim, saved_adapter_input_dim)
                            print(f"[DEBUG] Created projection layer: {current_clip_dim} -> {saved_adapter_input_dim}")
                        
                        # Recreate adapter with correct dimensions
                        mbart_dim = 1024
                        adapter_inputs = {
                            'clip_dim': saved_adapter_input_dim,
                            'mbart_dim': mbart_dim, 
                            'prefix_length': self.prefix_length
                        }
                        
                        if hasattr(self.adapter, 'num_encoder_layers'):
                            self.adapter = TransformerAdapter(num_encoder_layers=1, **adapter_inputs)
                        elif hasattr(self.adapter, 'hidden') and len(self.adapter.hidden) > 2:
                            self.adapter = HiddenMLPAdapter(**adapter_inputs)
                        else:
                            self.adapter = MLPAdapter(**adapter_inputs)
                        
                        print(f"[DEBUG] Recreated adapter with correct input dim: {saved_adapter_input_dim}")
                
                try:
                    missing_keys, unexpected_keys = self.adapter.load_state_dict(adapter_state_dict, strict=False)
                    print(f"[DEBUG] Successfully loaded adapter weights from CycleGAN checkpoint")
                    print(f"[DEBUG] Adapter missing keys: {len(missing_keys)}")
                    print(f"[DEBUG] Adapter unexpected keys: {len(unexpected_keys)}")
                    if missing_keys:
                        print(f"[DEBUG] Missing adapter keys: {missing_keys[:5]}...")
                except Exception as e:
                    print(f"[DEBUG] Error loading adapter weights from CycleGAN checkpoint: {e}")
            else:
                print(f"[DEBUG] No adapter.pt found in CycleGAN checkpoint: {adapter_file}")
            
            # mBART weights are already loaded by from_pretrained() from pytorch_model.bin
            print(f"[DEBUG] mBART weights loaded from pytorch_model.bin in: {checkpoint_path}")
            
            # Load projection layer weights if they exist
            projection_file = os.path.join(checkpoint_path, "clip_projection.pt")
            if self.clip_projection is not None and os.path.exists(projection_file):
                print(f"[DEBUG] Loading projection layer weights from: {projection_file}")
                projection_state_dict = torch.load(projection_file, map_location='cpu')
                try:
                    self.clip_projection.load_state_dict(projection_state_dict)
                    print(f"[DEBUG] Successfully loaded projection layer weights")
                except Exception as e:
                    print(f"[DEBUG] Error loading projection layer weights: {e}")
            elif self.clip_projection is not None:
                print(f"[DEBUG] Projection layer exists but no saved weights found - using random weights: {projection_file}")
            
        except Exception as e:
            print(f"[DEBUG] Error loading CycleGAN checkpoint: {e}")

    def train(self, mode: bool = True):
        """Set model in training mode"""
        super().train(mode)  # Call parent train method
        if mode:
            self.mbart.train()
            self.adapter.train()
            if self.clip_projection is not None:
                self.clip_projection.train()
        else:
            self.mbart.eval()
            self.adapter.eval()
            if self.clip_projection is not None:
                self.clip_projection.eval()
        # Keep M-CLIP in eval mode as it's frozen
        self.mclip.eval()
        return self

    def eval(self):
        """Set model in evaluation mode"""
        return self.train(False)
    
    def forward(
        self,
        sentences: List[str],
        target_sentences: List[str] = None,
        device = None,
        ):
        """
        Forward pass for the CLIPTrans generator.
        
        Args:
            sentences: List of source sentences
            target_sentences: List of target sentences (for training)
            device: Device to run computations on
            
        Returns:
            If target_sentences provided: (output_ids, transferred_sentences, supervised_loss)
            Else: (output_ids, transferred_sentences)
        """
        print(f"[DEBUG][CLIPTransGeneratorModel] forward() called with {len(sentences)} sentences.")
        # Extract features from source text using M-CLIP
        with torch.no_grad():
            visual_context = self.mclip(sentences)
        print(f"[DEBUG][CLIPTransGeneratorModel] visual_context shape: {visual_context.shape}")
        
        # Apply projection layer if needed for dimension mismatch
        if self.clip_projection is not None:
            visual_context = self.clip_projection(visual_context)
            print(f"[DEBUG][CLIPTransGeneratorModel] projected visual_context shape: {visual_context.shape}")
        
        # Transform features to decoder prefix using mapping network
        prefix_embeds = self.adapter(visual_context)
        print(f"[DEBUG][CLIPTransGeneratorModel] prefix_embeds shape: {prefix_embeds.shape}")
        # Tokenize input sentences for mBART
        inputs = self.tokenizer(sentences, 
            truncation=self.truncation, 
            padding=self.padding, 
            max_length=self.max_seq_length,
            return_tensors="pt")
        print(f"[DEBUG][CLIPTransGeneratorModel] input_ids shape: {inputs['input_ids'].shape}")
        
        if target_sentences is not None:
            # Training mode with target sentences
            labels = self.tokenizer(target_sentences,
                truncation=self.truncation, 
                padding=self.padding, 
                max_length=self.max_seq_length,
                return_tensors="pt").input_ids
            
            inputs = inputs.to(device)
            labels = labels.to(device)
            prefix_embeds = prefix_embeds.to(device)
            
            # Create decoder input embeddings with prefix
            decoder_input_ids = shift_tokens_right(labels, self.tokenizer.pad_token_id)
            decoder_inputs_embeds = self.mbart.model.decoder.embed_tokens(decoder_input_ids) * self.mbart.model.decoder.embed_scale
            decoder_inputs_embeds = torch.cat((decoder_inputs_embeds[:, 0].unsqueeze(1), prefix_embeds, decoder_inputs_embeds[:, 1:]), dim=1)
            
            # Set decoder attention mask
            decoder_attention_mask = torch.ones_like(decoder_input_ids).to(device)
            decoder_attention_mask[decoder_input_ids == self.tokenizer.pad_token_id] = 0
            decoder_attention_mask = torch.cat((decoder_attention_mask[:, 0].unsqueeze(1), 
                                              torch.ones(prefix_embeds.shape[0], self.prefix_length).to(device), 
                                              decoder_attention_mask[:, 1:]), dim=1)
            
            # Modify labels to accommodate prefix
            dummy_tokens = (torch.ones((labels.shape[0], self.prefix_length)) * self.tokenizer.pad_token_id).to(device)
            modified_labels = torch.cat((labels[:, 0].unsqueeze(1), dummy_tokens, labels[:, 1:]), dim=1).long()
            
            # Set padding tokens to -100 to ignore in loss calculation
            modified_labels[modified_labels == self.tokenizer.pad_token_id] = -100
            
            # Forward pass with prefix embeddings
            output_supervised = self.mbart(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                decoder_inputs_embeds=decoder_inputs_embeds,
                decoder_attention_mask=decoder_attention_mask,
                labels=modified_labels
            )
            
            # Generate output for cycle consistency (using CLIPTrans Stage 2 parameters)
            # For generation, use the inference format: prefix_embeds + EOS token
            gen_decoder_input_ids = (torch.ones(inputs.input_ids.shape[0], 1) * 2).long().to(device)  # EOS token
            gen_decoder_inputs_embeds = self.mbart.model.decoder.embed_tokens(gen_decoder_input_ids) * self.mbart.model.decoder.embed_scale
            gen_decoder_inputs_embeds = torch.cat((prefix_embeds, gen_decoder_inputs_embeds), dim=1)
            
            # Set decoder attention mask (CLIPTrans Stage 2 format)
            gen_decoder_attention_mask = torch.ones((gen_decoder_inputs_embeds.shape[0], gen_decoder_inputs_embeds.shape[1])).to(device)
            
            output = self.mbart.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                decoder_inputs_embeds=gen_decoder_inputs_embeds,
                decoder_attention_mask=gen_decoder_attention_mask,
                forced_bos_token_id=self.tokenizer.lang_code_to_id[self.target_lang_token],
                max_new_tokens=60
            )
            
            # Decode the output
            transferred_sentences = self.tokenizer.batch_decode(output, skip_special_tokens=True)
            
            return output, transferred_sentences, output_supervised.loss
        else:
            # Inference mode without target sentences
            inputs = inputs.to(device)
            prefix_embeds = prefix_embeds.to(device)
            
            # Create decoder input embeddings with prefix for generation (CLIPTrans Stage 2 format)
            # In CLIPTrans: prefix_embeds first, then EOS token (token 2)
            decoder_input_ids = (torch.ones(inputs.input_ids.shape[0], 1) * 2).long().to(device)  # EOS token
            decoder_inputs_embeds = self.mbart.model.decoder.embed_tokens(decoder_input_ids) * self.mbart.model.decoder.embed_scale
            decoder_inputs_embeds = torch.cat((prefix_embeds, decoder_inputs_embeds), dim=1)
            
            # Set decoder attention mask (CLIPTrans Stage 2 format)
            decoder_attention_mask = torch.ones((decoder_inputs_embeds.shape[0], decoder_inputs_embeds.shape[1])).to(device)
            
            # Generate output (using CLIPTrans Stage 2 parameters)
            output = self.mbart.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                decoder_inputs_embeds=decoder_inputs_embeds,
                decoder_attention_mask=decoder_attention_mask,
                forced_bos_token_id=self.tokenizer.lang_code_to_id[self.target_lang_token],
                max_new_tokens=60
            )
            
            # Decode the output
            transferred_sentences = self.tokenizer.batch_decode(output, skip_special_tokens=True)
            
            return output, transferred_sentences
        
    def transfer(
        self,
        sentences: List[str],
        device = None
        ):
        """
        Transfer text using the CLIPTrans generator.
        
        Args:
            sentences: List of source sentences
            device: Device to run computations on
            
        Returns:
            List[str]: Transferred sentences
        """
        # Extract features from source text using M-CLIP
        with torch.no_grad():
            visual_context = self.mclip(sentences)
        
        # Apply projection layer if needed for dimension mismatch
        if self.clip_projection is not None:
            visual_context = self.clip_projection(visual_context)
        
        # Transform features to decoder prefix using mapping network
        prefix_embeds = self.adapter(visual_context)
        
        # Tokenize input sentences for mBART
        inputs = self.tokenizer(sentences, 
            truncation=self.truncation, 
            padding=self.padding, 
            max_length=self.max_seq_length,
            return_tensors="pt")
        
        inputs = inputs.to(device)
        prefix_embeds = prefix_embeds.to(device)
        
        # Create decoder input embeddings with prefix for generation (CLIPTrans Stage 2 format)
        # In CLIPTrans: prefix_embeds first, then EOS token (token 2)
        decoder_input_ids = (torch.ones(inputs.input_ids.shape[0], 1) * 2).long().to(device)  # EOS token
        decoder_inputs_embeds = self.mbart.model.decoder.embed_tokens(decoder_input_ids) * self.mbart.model.decoder.embed_scale
        decoder_inputs_embeds = torch.cat((prefix_embeds, decoder_inputs_embeds), dim=1)
        
        # Set decoder attention mask (CLIPTrans Stage 2 format)
        decoder_attention_mask = torch.ones((decoder_inputs_embeds.shape[0], decoder_inputs_embeds.shape[1])).to(device)
        
        # Generate output (using CLIPTrans Stage 2 parameters)
        output = self.mbart.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            decoder_inputs_embeds=decoder_inputs_embeds,
            decoder_attention_mask=decoder_attention_mask,
            forced_bos_token_id=self.tokenizer.lang_code_to_id[self.target_lang_token],
            max_new_tokens=60
        )
        
        # Decode
        transferred_sentences = self.tokenizer.batch_decode(output, skip_special_tokens=True)

        return transferred_sentences
    
    def save_model(
        self, 
        path: Union[str]
        ):
        """
        Save the model components.
        
        Args:
            path: Directory path to save the model
        """
        os.makedirs(path, exist_ok=True)
        
        # Save mBART model and tokenizer
        self.mbart.save_pretrained(path)
        self.tokenizer.save_pretrained(f"{path}/tokenizer/")
        
        # Save adapter separately
        torch.save(self.adapter.state_dict(), f"{path}/adapter.pt")
        
        # Save projection layer if it exists
        if self.clip_projection is not None:
            torch.save(self.clip_projection.state_dict(), f"{path}/clip_projection.pt")
            print(f"[DEBUG] Saved projection layer weights to {path}/clip_projection.pt")
        
        print(f"Model saved to {path}") 