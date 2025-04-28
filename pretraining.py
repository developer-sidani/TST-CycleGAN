from comet_ml import Experiment, ExistingExperiment
from dotenv import load_dotenv

from cyclegan_tst.models.GeneratorModel import GeneratorModel
from data.datasets import ImageCaptionDataset
from utils.utils import *

import argparse
import logging
from tqdm import tqdm
import os, sys, time
import pickle
import numpy as np, pandas as pd
import random
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import get_scheduler
import torchvision.transforms as transforms

logging.basicConfig(level=logging.INFO)

SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

def parse_args():
    parser = argparse.ArgumentParser()
    
    # Basic parameters
    parser.add_argument('--image_dir', type=str, required=True, help='Directory containing the images')
    parser.add_argument('--caption_file', type=str, required=True, help='File containing image captions')
    parser.add_argument('--max_samples', type=int, default=None, help='Maximum number of samples to use')
    parser.add_argument('--lang', type=str, default='en', help='Language of the captions')
    
    # Model parameters
    parser.add_argument('--generator_model_tag', type=str, default='facebook/mbart-large-50', help='The tag of the model for the generator (e.g., "facebook/mbart-large-50")')
    parser.add_argument('--clip_model_name', type=str, default='openai/clip-vit-base-patch32', help='The tag of the CLIP model')
    parser.add_argument('--prefix_length', type=int, default=10, help='Length of the prefix for CLIP embeddings')
    parser.add_argument('--mapping_network', type=str, default='mlp', choices=['mlp', 'hidden_mlp', 'transformer'], help='Type of mapping network')
    parser.add_argument('--max_sequence_length', type=int, default=64, help='Max sequence length')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size used during training')
    parser.add_argument('--shuffle', action='store_true', default=True, help='Whether to shuffle the training set or not')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers used for dataloaders')
    parser.add_argument('--pin_memory', action='store_true', default=False, help='Whether to pin memory for data on GPU during data loading')
    parser.add_argument('--use_cuda_if_available', action='store_true', default=True, help='Whether to use GPU if available')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='Initial learning rate')
    parser.add_argument('--epochs', type=int, default=10, help='The number of training epochs')
    parser.add_argument('--lr_scheduler_type', type=str, default="linear", help='The scheduler used for the learning rate management')
    parser.add_argument('--warmup', action='store_true', default=False, help='Whether to apply warmup')
    
    # Saving and evaluation arguments
    parser.add_argument('--save_base_folder', type=str, required=True, help='The folder to use as base path to store model checkpoints')
    parser.add_argument('--from_pretrained', type=str, default=None, help='The folder to use as base path to load model checkpoints')
    parser.add_argument('--save_steps', type=int, default=1, help='How many training epochs between two checkpoints')
    parser.add_argument('--eval_steps', type=int, default=100, help='How many training steps between two evaluations')
    
    # Comet arguments
    parser.add_argument('--comet_logging', action='store_true', default=False, help='Set flag to enable comet logging')
    parser.add_argument('--comet_key', type=str, default=None, help='Comet API key to log some metrics')
    parser.add_argument('--comet_workspace', type=str, default=None, help='Comet workspace name')
    parser.add_argument('--comet_project_name', type=str, default=None, help='Comet experiment name')
    parser.add_argument('--comet_exp', type=str, default=None, help='Comet experiment key to continue logging')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Print arguments
    hyper_params = {}
    print("Arguments summary: \n")
    for key, value in vars(args).items():
        hyper_params[key] = value
        print(f"\t{key}:\t\t{value}")
    
    # Set device
    if args.use_cuda_if_available:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    else:
        device = torch.device("cpu")
    
    print(f"Using device: {device}")
    
    # Define image transformations
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Create dataset
    dataset = ImageCaptionDataset(
        image_dir=args.image_dir,
        caption_file=args.caption_file,
        transform=preprocess,
        max_dataset_samples=args.max_samples,
        lang=args.lang
    )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        collate_fn=ImageCaptionDataset.collate_fn
    )
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Number of batches: {len(dataloader)}")
    
    # Initialize model
    if args.from_pretrained is not None:
        model = GeneratorModel(
            model_name_or_path=args.generator_model_tag,
            pretrained_path=args.from_pretrained,
            max_seq_length=args.max_sequence_length,
            src_lang=args.lang,
            tgt_lang=args.lang,  # Same language for pretraining
            use_clip=True,
            clip_model_name=args.clip_model_name,
            prefix_length=args.prefix_length,
            mapping_network=args.mapping_network
        )
        print('Generator pretrained model loaded correctly')
    else:
        model = GeneratorModel(
            model_name_or_path=args.generator_model_tag,
            max_seq_length=args.max_sequence_length,
            src_lang=args.lang,
            tgt_lang=args.lang,  # Same language for pretraining
            use_clip=True,
            clip_model_name=args.clip_model_name,
            prefix_length=args.prefix_length,
            mapping_network=args.mapping_network
        )
        print('Generator pretrained model not loaded - Initial weights will be used')
    
    # Move model to device
    model.model.to(device)
    if hasattr(model, 'adapter'):
        model.adapter.to(device)
    
    # Setup optimizer and scheduler
    num_training_steps = args.epochs * len(dataloader)
    print(f"Total number of training steps: {num_training_steps}")
    
    warmup_steps = int(0.1 * num_training_steps) if args.warmup else 0
    
    # Collect all parameters that require gradients
    optimizer_params = list(model.model.parameters())
    if hasattr(model, 'adapter'):
        optimizer_params += list(model.adapter.parameters())
    
    optimizer = AdamW(optimizer_params, lr=args.learning_rate)
    lr_scheduler = get_scheduler(
        args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps
    )
    
    start_epoch = 0
    current_training_step = 0
    
    if args.from_pretrained is not None and os.path.exists(f"{args.from_pretrained}checkpoint.pth"):
        checkpoint = torch.load(f"{args.from_pretrained}checkpoint.pth", map_location=torch.device("cpu"))
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        start_epoch = checkpoint['epoch']
        current_training_step = checkpoint['training_step']
        del checkpoint
    
    # Setup Comet logging
    if args.comet_logging:
        if args.from_pretrained is not None and args.comet_exp is not None:
            experiment = ExistingExperiment(api_key=args.comet_key, previous_experiment=args.comet_exp)
        else:
            load_dotenv()
            comet_api_key = os.getenv('COMET_API_KEY') if args.comet_key is None else args.comet_key
            comet_project = os.getenv('COMET_PROJECT_NAME') if args.comet_project_name is None else args.comet_project_name
            comet_workspace = os.getenv('COMET_WORKSPACE') if args.comet_workspace is None else args.comet_workspace
            experiment = Experiment(
                api_key=comet_api_key,
                project_name=comet_project,
                workspace=comet_workspace,
            )
            experiment.set_name(f"pretraining_clip_to_text_{args.lang}")
        experiment.log_parameters(hyper_params)
    else:
        experiment = None
    
    # Training loop
    progress_bar = tqdm(range(num_training_steps))
    progress_bar.update(current_training_step)
    
    loss_logging = {'Loss': []}
    loss_logging['hyper_params'] = hyper_params
    
    print('Start pretraining...')
    for epoch in range(start_epoch, args.epochs):
        print(f"\nPretraining epoch: {epoch}")
        model.train()  # Set training mode
        
        epoch_loss = 0.0
        
        for batch_idx, (images, captions) in enumerate(dataloader):
            # Forward pass
            _, _, loss = model.forward_with_clip(images, captions, device=device)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            
            # Logging
            loss_value = loss.item()
            epoch_loss += loss_value
            loss_logging['Loss'].append(loss_value)
            
            if experiment is not None:
                with experiment.train():
                    experiment.log_metric('loss', loss_value, step=current_training_step)
                    experiment.log_metric('learning_rate', lr_scheduler.get_last_lr()[0], step=current_training_step)
            
            # Update progress bar
            progress_bar.update(1)
            current_training_step += 1
            
            # Evaluation
            if batch_idx > 0 and batch_idx % args.eval_steps == 0:
                model.eval()
                # Get a few example generations
                with torch.no_grad():
                    sample_images = images[:2] if len(images) >= 2 else images
                    sample_captions = captions[:2] if len(captions) >= 2 else captions
                    generated_ids, generated_texts = model.forward_with_clip(sample_images, device=device)
                
                if experiment is not None:
                    with experiment.test():
                        for i, (image, caption, generated) in enumerate(zip(sample_images[:2], sample_captions[:2], generated_texts[:2])):
                            experiment.log_text(f"Example {i+1}:\nActual: {caption}\nGenerated: {generated}", step=current_training_step)
                            if hasattr(image, 'cpu'):
                                # Convert tensor to PIL image for logging
                                image_pil = transforms.ToPILImage()(image.cpu())
                                experiment.log_image(image_pil, name=f"example_{i+1}", step=current_training_step)
                            
                model.train()
        
        # End of epoch
        epoch_loss /= len(dataloader)
        print(f"Epoch {epoch} - Average Loss: {epoch_loss:.4f}")
        
        if experiment is not None:
            with experiment.train():
                experiment.log_metric('epoch_loss', epoch_loss, step=epoch)
        
        # Save checkpoint
        if (epoch + 1) % args.save_steps == 0:
            save_path = os.path.join(args.save_base_folder, f"epoch_{epoch+1}")
            os.makedirs(save_path, exist_ok=True)
            
            model.save_model(save_path)
            
            checkpoint = {
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch + 1,
                'training_step': current_training_step
            }
            
            torch.save(checkpoint, os.path.join(save_path, "checkpoint.pth"))
            print(f"Model saved to {save_path}")
    
    # Save final model
    final_path = os.path.join(args.save_base_folder, "final")
    os.makedirs(final_path, exist_ok=True)
    
    model.save_model(final_path)
    
    checkpoint = {
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict(),
        'epoch': args.epochs,
        'training_step': current_training_step
    }
    
    torch.save(checkpoint, os.path.join(final_path, "checkpoint.pth"))
    
    # Save loss logs
    with open(os.path.join(args.save_base_folder, "loss_logs.pkl"), 'wb') as f:
        pickle.dump(loss_logging, f)
    
    print("Pretraining completed!")

if __name__ == "__main__":
    main()