from comet_ml import Experiment

from data.datasets import MonostyleDataset, ParallelRefDataset, ImageCaptionDataset
from cyclegan_tst.models.CycleGANModel import CycleGANModel
from cyclegan_tst.models.DiscriminatorModel import DiscriminatorModel
from cyclegan_tst.models.GeneratorModel import GeneratorModel
from eval import *
from utils.utils import *

import argparse
import logging
import os
import numpy as np, pandas as pd
import random
from PIL import Image
import torchvision.transforms as transforms

import torch
from torch.utils.data import DataLoader, Dataset

logging.basicConfig(level=logging.INFO)

SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

class FileInputDataset(Dataset):
    """Dataset for loading inputs from a file."""
    def __init__(self, file_path, input_mode="text", transform=None):
        self.input_mode = input_mode
        self.transform = transform
        
        with open(file_path, 'r', encoding='utf-8') as f:
            self.lines = [line.strip() for line in f.readlines()]
            
        # If mode is image, lines should be paths to image files
        if input_mode == "image":
            self.image_paths = self.lines
            # Check if images exist
            for img_path in self.image_paths:
                if not os.path.exists(img_path):
                    logging.warning(f"Image file not found: {img_path}")
    
    def __len__(self):
        return len(self.lines)
    
    def __getitem__(self, idx):
        if self.input_mode == "text":
            return self.lines[idx]
        else:  # image
            try:
                image = Image.open(self.image_paths[idx]).convert('RGB')
                if self.transform:
                    image = self.transform(image)
                return image
            except Exception as e:
                logging.error(f"Error loading image {self.image_paths[idx]}: {e}")
                # Return a placeholder black image
                placeholder = Image.new('RGB', (224, 224), color='black')
                if self.transform:
                    placeholder = self.transform(placeholder)
                return placeholder
    
    @staticmethod
    def collate_fn_text(batch):
        return batch
    
    @staticmethod
    def collate_fn_image(batch):
        return batch

''' 
    ----- ----- ----- ----- ----- ----- ----- -----
                    PARSING PARAMs       
    ----- ----- ----- ----- ----- ----- ----- -----
'''

parser = argparse.ArgumentParser()

# basic parameters
parser.add_argument('--style_a', type=str, dest="style_a", help='style A for the style transfer task (source style for G_ab).')
parser.add_argument('--style_b', type=str, dest="style_b", help='style B for the style transfer task (target style for G_ab).')
parser.add_argument('--lang', type=str, dest="lang", default='en', help='Dataset language.')
parser.add_argument('--direction', type=str, choices=['AB', 'BA'], default='AB', help='Direction of transfer: AB (A->B) or BA (B->A)')
parser.add_argument('--input_mode', type=str, choices=['text', 'image'], default='text', help='Input mode: text or image')
parser.add_argument('--input_file', type=str, required=True, help='Path to input file (text lines or image paths)')
parser.add_argument('--output_file', type=str, required=True, help='Path to output file for results')

# CLIP arguments
parser.add_argument('--use_clip', action='store_true', default=False, help='Whether to use CLIP for image input')
parser.add_argument('--clip_model_name', type=str, default='openai/clip-vit-base-patch32', help='The tag of the CLIP model')
parser.add_argument('--prefix_length', type=int, default=10, help='Length of the prefix for CLIP embeddings')
parser.add_argument('--mapping_network', type=str, default='mlp', 
                   choices=['mlp', 'hidden_mlp', 'transformer'], help='Type of mapping network for CLIP')

# Keep existing test dataset arguments for compatibility
parser.add_argument('--max_samples_test',  type=int, dest="max_samples_test",  default=None, help='Max number of examples to retain from the test set. None for all available examples.')
parser.add_argument('--test_ds', type=str, dest="test_ds", default='custom', help='Test dataset.')
parser.add_argument('--path_mono_A_test', type=str, dest="path_mono_A_test", help='Path to non-parallel dataset (style A) for test.')
parser.add_argument('--path_mono_B_test', type=str, dest="path_mono_B_test", help='Path to non-parallel dataset (style B) for test.')
parser.add_argument('--path_paral_A_test', type=str, dest="path_paral_A_test", help='Path to parallel dataset (style A) for test.')
parser.add_argument('--path_paral_B_test', type=str, dest="path_paral_B_test", help='Path to parallel dataset (style B) for test.')
parser.add_argument('--path_paral_test_ref', type=str, dest="path_paral_test_ref", help='Path to human references for test.')
parser.add_argument('--n_references',  type=int, dest="n_references",  default=None, help='Number of human references for test.')
parser.add_argument('--lowercase_ref', action='store_true', dest="lowercase_ref", default=False, help='Whether to lowercase references.')
parser.add_argument('--bertscore', action='store_true', dest="bertscore", default=True, help='Whether to compute BERTScore metric.')

parser.add_argument('--max_sequence_length', type=int,  dest="max_sequence_length", default=64, help='Max sequence length')

# Batch processing arguments
parser.add_argument('--batch_size', type=int,  dest="batch_size",  default=64,     help='Batch size used during inference.')
parser.add_argument('--num_workers',type=int,  dest="num_workers", default=4,     help='Number of workers used for dataloaders.')
parser.add_argument('--pin_memory', action='store_true', dest="pin_memory",  default=False, help='Whether to pin memory for data on GPU during data loading.')
parser.add_argument('--use_cuda_if_available', action='store_true', dest="use_cuda_if_available", default=False, help='Whether to use GPU if available.')

# Model arguments
parser.add_argument('--generator_model_tag', type=str, dest="generator_model_tag", default="facebook/mbart-large-50", help='The tag of the model for the generator (e.g., "facebook/mbart-large-50").')
parser.add_argument('--discriminator_model_tag', type=str, dest="discriminator_model_tag", default="distilbert-base-cased", help='The tag of the model discriminator (e.g., "distilbert-base-cased").')
parser.add_argument('--model_path', type=str, required=True, help='Path to the model for inference')

# args for comet
parser.add_argument('--comet_logging', action='store_true', dest="comet_logging", default=False, help='Set flag to enable comet logging')
parser.add_argument('--comet_key', type=str, dest="comet_key", default=None, help='Comet API key to log some metrics')
parser.add_argument('--comet_workspace', type=str, dest="comet_workspace", default=None, help='Comet workspace name')
parser.add_argument('--comet_project_name', type=str, dest="comet_project_name", default=None, help='Comet experiment name')

args = parser.parse_args()

hyper_params = {}
print("Arguments summary:\n")
for key, value in vars(args).items():
    hyper_params[key] = value
    print(f"\t{key}:\t\t{value}")

# Initialize Comet if needed
if args.comet_logging:
    from dotenv import load_dotenv
    load_dotenv()
    args.comet_key = os.getenv('COMET_API_KEY') if args.comet_key is None else args.comet_key
    args.comet_workspace = os.getenv('COMET_WORKSPACE') if args.comet_workspace is None else args.comet_workspace
    args.comet_project_name = os.getenv('COMET_PROJECT_NAME') if args.comet_project_name is None else args.comet_project_name
    experiment = Experiment(api_key=args.comet_key,
                           project_name=args.comet_project_name,
                           workspace=args.comet_workspace)
    experiment.set_name(f"inference_{args.style_a}_to_{args.style_b}_{args.input_mode}")
    experiment.log_parameters(hyper_params)
else:
    experiment = None

# Set device
if args.use_cuda_if_available:
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

# Load the dataset from the input file
if args.input_mode == "image" and args.use_clip:
    # For image input, we need to apply transforms
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_dataset = FileInputDataset(args.input_file, input_mode="image", transform=preprocess)
    collate_fn = FileInputDataset.collate_fn_image
else:
    # For text input
    input_dataset = FileInputDataset(args.input_file, input_mode="text")
    collate_fn = FileInputDataset.collate_fn_text

input_dataloader = DataLoader(
    input_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=args.num_workers,
    pin_memory=args.pin_memory,
    collate_fn=collate_fn
)

print(f"Input dataset size: {len(input_dataset)}")
print(f"Number of batches: {len(input_dataloader)}")

# Load models
src_lang = args.style_a if args.direction == 'AB' else args.style_b
tgt_lang = args.style_b if args.direction == 'AB' else args.style_a

print(f"Loading model for {src_lang} to {tgt_lang} transfer...")

# Load generator model
if args.input_mode == "image" and args.use_clip:
    generator = GeneratorModel(
        args.generator_model_tag,
        f'{args.model_path}G_{args.direction.lower()}/',
        max_seq_length=args.max_sequence_length,
        src_lang=src_lang,
        tgt_lang=tgt_lang,
        use_clip=True,
        clip_model_name=args.clip_model_name,
        prefix_length=args.prefix_length,
        mapping_network=args.mapping_network
    )
else:
    generator = GeneratorModel(
        args.generator_model_tag,
        f'{args.model_path}G_{args.direction.lower()}/',
        max_seq_length=args.max_sequence_length,
        src_lang=src_lang,
        tgt_lang=tgt_lang
    )

generator.model.to(device)
if hasattr(generator, 'adapter') and args.use_clip:
    generator.adapter.to(device)

generator.eval()  # Set to evaluation mode

# Run inference
print("Running inference...")
all_outputs = []

with torch.no_grad():
    for batch in input_dataloader:
        if args.input_mode == "image" and args.use_clip:
            # Process images through the CLIP-based model
            outputs = generator.transfer(images=batch, device=device)
        else:
            # Process text
            outputs = generator.transfer(sentences=batch, device=device)
        
        all_outputs.extend(outputs)

# Save results to output file
with open(args.output_file, 'w', encoding='utf-8') as f:
    for output in all_outputs:
        f.write(f"{output}\n")

print(f"Inference completed. Results saved to {args.output_file}")

# Log a few examples if using Comet
if experiment is not None:
    num_examples = min(5, len(all_outputs))
    for i in range(num_examples):
        input_text = input_dataset.lines[i]
        output_text = all_outputs[i]
        experiment.log_text(f"Example {i+1}:\nInput: {input_text}\nOutput: {output_text}")
        
        if args.input_mode == "image" and os.path.exists(input_text):
            try:
                image = Image.open(input_text)
                experiment.log_image(image, name=f"input_image_{i+1}")
            except Exception as e:
                logging.error(f"Could not log image: {e}")

print("Done.")
