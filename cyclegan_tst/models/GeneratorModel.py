from typing import List, Optional, Tuple, Union
from torch import nn
import torch

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, MBartForConditionalGeneration, MBart50TokenizerFast
from tqdm import tqdm
from utils.utils import get_lang_code

class GeneratorModel(nn.Module):
    
    def __init__(
        self,
        model_name_or_path: str,
        pretrained_path: str = None,
        max_seq_length: int = 64,
        truncation: str = "longest_first",
        padding: str = "max_length",
        src_lang: str = "en",  # Short language code
        tgt_lang: str = "de"   # Short language code
        ):
        super(GeneratorModel, self).__init__()
        
        self.model_name_or_path = model_name_or_path
        self.max_seq_length = max_seq_length
        self.truncation = truncation
        self.padding = padding
        
        # First, check if this is likely an mBART model
        self.is_mbart = self._is_mbart_model_by_name()
        
        # Load model and tokenizer based on detection
        if pretrained_path is None:
            if self.is_mbart:
                self.model = MBartForConditionalGeneration.from_pretrained(model_name_or_path)
                self.tokenizer = MBart50TokenizerFast.from_pretrained(model_name_or_path)
            else:
                self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
                self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        else:
            if self.is_mbart:
                self.model = MBartForConditionalGeneration.from_pretrained(pretrained_path)
                self.tokenizer = MBart50TokenizerFast.from_pretrained(f"{pretrained_path}tokenizer/")
            else:
                self.model = AutoModelForSeq2SeqLM.from_pretrained(pretrained_path)
                self.tokenizer = AutoTokenizer.from_pretrained(f"{pretrained_path}tokenizer/")
        
        # Print model information for debugging
        print(f"Loaded model: {type(self.model).__name__}")
        print(f"Model type: {getattr(self.model.config, 'model_type', 'Unknown')}")
        print(f"Tokenizer type: {type(self.tokenizer).__name__}")
        print(f"Detected as mBART: {self.is_mbart}")
        
        if self.is_mbart:
            # Convert short language codes to full mBART language codes
            self.src_lang = get_lang_code(src_lang)
            self.tgt_lang = get_lang_code(tgt_lang)
            
            print(f"mBART language codes - Source: {self.src_lang}, Target: {self.tgt_lang}")
            
            # Ensure tokenizer has the required language codes
            if not hasattr(self.tokenizer, 'lang_code_to_id'):
                raise ValueError("mBART tokenizer missing lang_code_to_id mapping")
        else:
            # For non-mBART models, keep the original language codes
            self.src_lang = src_lang
            self.tgt_lang = tgt_lang
            
            print(f"Generic language codes - Source: {self.src_lang}, Target: {self.tgt_lang}")
    
    def _is_mbart_model_by_name(self) -> bool:
        """
        Check if the model is likely mBART based on the model name/path.
        This is done before loading to decide which classes to use.
        """
        model_name = self.model_name_or_path.lower()
        mbart_indicators = ['mbart', 'facebook/mbart']
        return any(indicator in model_name for indicator in mbart_indicators)

    def train(self):
        # Setting the model in training mode
        self.model.train()

    def eval(self):
        # Setting the model in evaluation mode
        self.model.eval()
    
    def forward(
        self,
        sentences: List[str],
        target_sentences: List[str] = None,
        device = None,
        ):
        # Tokenize input sentences
        inputs = self.tokenizer(sentences, 
            truncation=self.truncation, 
            padding=self.padding, 
            max_length=self.max_seq_length,
            return_tensors="pt")

        if target_sentences is not None:
            labels = self.tokenizer(target_sentences,
                truncation=self.truncation, 
                padding=self.padding, 
                max_length=self.max_seq_length,
                return_tensors="pt").input_ids
            
            inputs = inputs.to(device)
            labels = labels.to(device)
            output_supervised = self.model(**inputs, labels=labels)
        
        inputs = inputs.to(device)
        
        # Use simple forward pass instead of generate() to avoid bugs in custom transformers
        # This implements a basic greedy decoding approach
        if self.is_mbart:
            # For mBART models, we need to set the decoder_start_token_id
            decoder_start_token_id = self.tokenizer.lang_code_to_id[self.tgt_lang]
        else:
            # For non-mBART models, use the model's default
            decoder_start_token_id = getattr(self.model.config, 'decoder_start_token_id', self.tokenizer.bos_token_id)
            if decoder_start_token_id is None:
                decoder_start_token_id = self.tokenizer.bos_token_id
        
        # Simple generation using model forward pass
        batch_size = inputs['input_ids'].shape[0]
        
        # Create initial decoder input
        if decoder_start_token_id is not None:
            decoder_input_ids = torch.full((batch_size, 1), decoder_start_token_id, device=device, dtype=torch.long)
        else:
            # Fallback to using the input structure
            decoder_input_ids = torch.full((batch_size, 1), self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0, device=device, dtype=torch.long)
        
        # Simple greedy generation loop
        for _ in range(self.max_seq_length - 1):
            # Forward pass
            with torch.no_grad():
                model_inputs = {**inputs, 'decoder_input_ids': decoder_input_ids}
                outputs = self.model(**model_inputs)
                
                # Get next token predictions
                next_token_logits = outputs.logits[:, -1, :]
                next_tokens = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
                
                # Concatenate with previous tokens
                decoder_input_ids = torch.cat([decoder_input_ids, next_tokens], dim=-1)
                
                # Check for end tokens (simple stopping criteria)
                if self.tokenizer.eos_token_id is not None:
                    if (next_tokens == self.tokenizer.eos_token_id).all():
                        break
        
        output = decoder_input_ids

        # Decode the output
        transferred_sentences = self.tokenizer.batch_decode(output, skip_special_tokens=True)

        if target_sentences is not None:
            return output, transferred_sentences, output_supervised.loss
        else:
            return output, transferred_sentences
        
    def transfer(
        self,
        sentences: List[str],
        device = None
        ):
        inputs = self.tokenizer(sentences, 
            truncation=self.truncation, 
            padding=self.padding, 
            max_length=self.max_seq_length,
            return_tensors="pt")
                
        inputs = inputs.to(device)
        
        # Use simple forward pass instead of generate() to avoid bugs in custom transformers
        # This implements a basic greedy decoding approach
        if self.is_mbart:
            # For mBART models, we need to set the decoder_start_token_id
            decoder_start_token_id = self.tokenizer.lang_code_to_id[self.tgt_lang]
        else:
            # For non-mBART models, use the model's default
            decoder_start_token_id = getattr(self.model.config, 'decoder_start_token_id', self.tokenizer.bos_token_id)
            if decoder_start_token_id is None:
                decoder_start_token_id = self.tokenizer.bos_token_id
        
        # Simple generation using model forward pass
        batch_size = inputs['input_ids'].shape[0]
        
        # Create initial decoder input
        if decoder_start_token_id is not None:
            decoder_input_ids = torch.full((batch_size, 1), decoder_start_token_id, device=device, dtype=torch.long)
        else:
            # Fallback to using the input structure
            decoder_input_ids = torch.full((batch_size, 1), self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0, device=device, dtype=torch.long)
        
        # Simple greedy generation loop
        for _ in range(self.max_seq_length - 1):
            # Forward pass
            with torch.no_grad():
                model_inputs = {**inputs, 'decoder_input_ids': decoder_input_ids}
                outputs = self.model(**model_inputs)
                
                # Get next token predictions
                next_token_logits = outputs.logits[:, -1, :]
                next_tokens = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
                
                # Concatenate with previous tokens
                decoder_input_ids = torch.cat([decoder_input_ids, next_tokens], dim=-1)
                
                # Check for end tokens (simple stopping criteria)
                if self.tokenizer.eos_token_id is not None:
                    if (next_tokens == self.tokenizer.eos_token_id).all():
                        break
        
        output = decoder_input_ids
        
        # Decode
        transferred_sentences = self.tokenizer.batch_decode(output, skip_special_tokens=True)

        return transferred_sentences
    
    def save_model(
        self, 
        path: Union[str]
        ):

        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(f"{path}/tokenizer/")

    def get_model_info(self) -> dict:
        """
        Returns information about the loaded model for debugging purposes.
        
        Returns:
            dict: Dictionary containing model information
        """
        return {
            'model_class': type(self.model).__name__,
            'model_type': getattr(self.model.config, 'model_type', 'Unknown'),
            'tokenizer_class': type(self.tokenizer).__name__,
            'is_mbart': self.is_mbart,
            'src_lang': self.src_lang,
            'tgt_lang': self.tgt_lang,
            'model_name_or_path': self.model_name_or_path
        }
