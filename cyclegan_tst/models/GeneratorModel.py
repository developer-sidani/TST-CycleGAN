from typing import List, Optional, Tuple, Union
from torch import nn

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
        
        # Convert short language codes to full mBART language codes
        self.src_lang = get_lang_code(src_lang)
        self.tgt_lang = get_lang_code(tgt_lang)
        
        if pretrained_path is None:
            self.model = MBartForConditionalGeneration.from_pretrained(model_name_or_path)
            self.tokenizer = MBart50TokenizerFast.from_pretrained(model_name_or_path)
        else:
            self.model = MBartForConditionalGeneration.from_pretrained(pretrained_path)
            self.tokenizer = MBart50TokenizerFast.from_pretrained(f"{pretrained_path}tokenizer/")


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
        output = self.model.generate(
            **inputs, 
            forced_bos_token_id=self.tokenizer.lang_code_to_id[self.tgt_lang],
            max_length=self.max_seq_length
        )

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

        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(f"{path}/tokenizer/")
