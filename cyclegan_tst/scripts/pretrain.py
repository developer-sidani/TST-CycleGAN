"""
Stage 1 pretraining script for Multimodal CycleGAN NMT.
Trains on image captioning to learn the alignment between visual and textual spaces.
"""

import os
import argparse
import logging
import pickle
import random
import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_scheduler
from torch.cuda.amp import autocast, GradScaler
import sacrebleu
import evaluate

# Import custom modules
from cyclegan_tst.models.MultimodalGeneratorModel import MultimodalGeneratorModel
from cyclegan_tst.utils.multimodal_data_utils import Multi30kDataLoader, MultimodalDataset, multimodal_collate_fn

try:
    from comet_ml import Experiment

    has_comet = True
except ImportError:
    has_comet = False

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# For reproducibility
SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Pretraining script for Multimodal CycleGAN NMT")

    # Data arguments
    parser.add_argument("--data_dir", type=str, default="data/multi30k",
                        help="Directory containing Multi30k data")
    parser.add_argument("--image_dir", type=str, default="data/multi30k/images",
                        help="Directory containing Multi30k images")
    parser.add_argument("--cache_dir", type=str, default="data/multi30k/cache",
                        help="Directory to cache processed data")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save model")
    parser.add_argument("--src_lang", type=str, default="en",
                        help="Source language")
    parser.add_argument("--tgt_lang", type=str, default="de",
                        help="Target language")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum number of samples to use (for debugging)")

    # Model arguments
    parser.add_argument("--mbart_model", type=str, default="facebook/mbart-large-50",
                        help="mBART model to use")
    parser.add_argument("--clip_model", type=str, default="M-CLIP/XLM-Roberta-Large-Vit-B-32",
                        help="CLIP model to use")
    parser.add_argument("--adapter_type", type=str, default="mlp",
                        choices=["mlp", "hidden_mlp", "transformer"],
                        help="Type of adapter to use")
    parser.add_argument("--prefix_length", type=int, default=10,
                        help="Length of prefix for adapter")
    parser.add_argument("--max_seq_length", type=int, default=64,
                        help="Maximum sequence length")

    # Training arguments
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of updates steps to accumulate before backward pass")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                        help="Initial learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay")
    parser.add_argument("--num_epochs", type=int, default=10,
                        help="Number of epochs")
    parser.add_argument("--warmup_ratio", type=float, default=0.1,
                        help="Ratio of steps for warmup")
    parser.add_argument("--logging_steps", type=int, default=100,
                        help="Log every X steps")
    parser.add_argument("--save_steps", type=int, default=1000,
                        help="Save checkpoint every X steps")
    parser.add_argument("--eval_steps", type=int, default=500,
                        help="Evaluate every X steps")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="Resume from checkpoint")
    parser.add_argument("--fp16", action="store_true",
                        help="Use mixed precision training")

    # Comet ML arguments
    parser.add_argument("--use_comet", action="store_true",
                        help="Whether to use Comet ML for logging")
    parser.add_argument("--comet_project", type=str, default="multimodal-cyclegan-nmt",
                        help="Comet ML project name")
    parser.add_argument("--comet_workspace", type=str, default=None,
                        help="Comet ML workspace")

    return parser.parse_args()


def main():
    """Main training function"""
    # Parse arguments
    args = parse_args()

    # Load environment variables for Comet ML
    if args.use_comet:
        load_dotenv()

    # Setup logging
    os.makedirs(args.output_dir, exist_ok=True)
    logging_file = os.path.join(args.output_dir, "training.log")
    file_handler = logging.FileHandler(logging_file)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    # Setup Comet ML
    if args.use_comet and has_comet:
        comet_experiment = Experiment(
            project_name=args.comet_project,
            workspace=args.comet_workspace
        )
        comet_experiment.log_parameters(vars(args))
    else:
        comet_experiment = None

    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Initialize data loader
    logger.info("Loading Multi30k data...")
    data_loader = Multi30kDataLoader(
        data_dir=args.data_dir,
        langs=[args.src_lang, args.tgt_lang],
        test_sets=[("2016", "flickr")],
        image_dir=args.image_dir,
        cache_dir=args.cache_dir,
        max_samples=args.max_samples
    )

    # Get train and test data
    logger.info("Creating datasets...")
    train_src, train_tgt, train_images = data_loader.get_train_data(
        src_lang=args.src_lang,
        tgt_lang=args.tgt_lang,
        include_images=True
    )

    test_src, test_tgt, test_images = data_loader.get_test_data(
        test_set=("2016", "flickr"),
        src_lang=args.src_lang,
        tgt_lang=args.tgt_lang,
        include_images=True
    )

    # Create datasets
    train_dataset = MultimodalDataset(train_src, train_tgt, train_images)
    test_dataset = MultimodalDataset(test_src, test_tgt, test_images)

    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=multimodal_collate_fn
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=multimodal_collate_fn
    )

    # Initialize model
    logger.info(f"Initializing model with {args.adapter_type} adapter...")
    model = MultimodalGeneratorModel(
        mbart_model_path=args.mbart_model,
        clip_model_path=args.clip_model,
        adapter_type=args.adapter_type,
        prefix_length=args.prefix_length,
        max_seq_length=args.max_seq_length,
        src_lang=args.src_lang,
        tgt_lang=args.tgt_lang,
        pretrained_path=args.resume_from_checkpoint
    )

    # Move model to device
    model.mbart_model.to(device)
    model.adapter.to(device)

    # Setup optimizer and scheduler
    optimizer = AdamW(
        [
            {"params": model.mbart_model.parameters()},
            {"params": model.adapter.parameters(), "lr": args.learning_rate * 5}  # Higher LR for adapter
        ],
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )

    num_training_steps = args.num_epochs * len(train_dataloader)
    num_warmup_steps = int(num_training_steps * args.warmup_ratio)

    scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )

    # Setup mixed precision training
    scaler = GradScaler() if args.fp16 else None

    # Initialize METEOR metric
    meteor = evaluate.load('meteor')

    # Training loop
    logger.info("Starting training...")
    global_step = 0

    for epoch in range(args.num_epochs):
        model.train()
        epoch_loss = 0

        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{args.num_epochs}")

        for step, batch in enumerate(progress_bar):
            # Get batch data
            src_texts = batch["src_texts"]
            tgt_texts = batch["tgt_texts"]
            images = batch["images"].to(device) if "images" in batch else None

            # Process images with CLIP
            with torch.no_grad():
                image_embs = model.encode_image(images, device=device)

            # Forward pass
            with autocast(enabled=args.fp16):
                # Generate captions using images
                _, generated_captions, caption_loss = model(
                    image_embs,
                    is_clip_embedding=True,
                    reference_sentences=tgt_texts,
                    device=device
                )

                # Apply gradient accumulation
                caption_loss = caption_loss / args.gradient_accumulation_steps

            # Backward pass with mixed precision
            if args.fp16:
                scaler.scale(caption_loss).backward()
            else:
                caption_loss.backward()

            # Update model parameters after gradient accumulation
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

                scheduler.step()
                optimizer.zero_grad()

                # Log training metrics
                if global_step % args.logging_steps == 0:
                    progress_bar.set_postfix({"loss": caption_loss.item() * args.gradient_accumulation_steps})

                    if comet_experiment is not None:
                        comet_experiment.log_metric(
                            "train_loss",
                            caption_loss.item() * args.gradient_accumulation_steps,
                            step=global_step
                        )

                # Save checkpoint
                if global_step % args.save_steps == 0:
                    checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    model.save_model(checkpoint_dir)
                    logger.info(f"Saved checkpoint to {checkpoint_dir}")

                # Evaluate model
                if global_step % args.eval_steps == 0:
                    eval_metrics = evaluate_model(model, test_dataloader, device, args.fp16)

                    logger.info(f"Eval metrics at step {global_step}: {eval_metrics}")

                    if comet_experiment is not None:
                        for metric_name, metric_value in eval_metrics.items():
                            comet_experiment.log_metric(f"eval_{metric_name}", metric_value, step=global_step)

                global_step += 1

            epoch_loss += caption_loss.item() * args.gradient_accumulation_steps

        # End of epoch
        avg_epoch_loss = epoch_loss / len(train_dataloader)
        logger.info(f"Epoch {epoch + 1}/{args.num_epochs} - Avg Loss: {avg_epoch_loss:.4f}")

        if comet_experiment is not None:
            comet_experiment.log_metric("epoch_loss", avg_epoch_loss, step=epoch)

        # Evaluate at the end of each epoch
        eval_metrics = evaluate_model(model, test_dataloader, device, args.fp16)
        logger.info(f"Epoch {epoch + 1}/{args.num_epochs} - Eval metrics: {eval_metrics}")

        if comet_experiment is not None:
            for metric_name, metric_value in eval_metrics.items():
                comet_experiment.log_metric(f"epoch_{metric_name}", metric_value, step=epoch)

        # Save model at the end of each epoch
        epoch_dir = os.path.join(args.output_dir, f"epoch-{epoch + 1}")
        os.makedirs(epoch_dir, exist_ok=True)
        model.save_model(epoch_dir)
        logger.info(f"Saved model for epoch {epoch + 1} to {epoch_dir}")

    # Save final model
    final_dir = os.path.join(args.output_dir, "final_model")
    os.makedirs(final_dir, exist_ok=True)
    model.save_model(final_dir)
    logger.info(f"Saved final model to {final_dir}")

    # End Comet experiment
    if comet_experiment is not None:
        comet_experiment.end()


def evaluate_model(model, dataloader, device, fp16=False):
    """Evaluate model on dataloader"""
    model.eval()
    all_references = []
    all_predictions = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Get batch data
            src_texts = batch["src_texts"]
            tgt_texts = batch["tgt_texts"]
            images = batch["images"].to(device) if "images" in batch else None

            # Process images with CLIP
            image_embs = model.encode_image(images, device=device)

            # Generate captions
            with autocast(enabled=fp16):
                _, generated_captions = model(
                    image_embs,
                    is_clip_embedding=True,
                    device=device,
                    generate_only=True
                )

            all_predictions.extend(generated_captions)
            all_references.extend(tgt_texts)

    # Calculate metrics
    bleu = sacrebleu.corpus_bleu(all_predictions, [all_references]).score
    meteor = evaluate.load('meteor')
    meteor_score = meteor.compute(predictions=all_predictions, references=all_references)['meteor']

    metrics = {
        "bleu": bleu,
        "meteor": meteor_score * 100  # Convert to percentage
    }

    # Log sample predictions
    num_samples = min(5, len(all_predictions))
    for i in range(num_samples):
        logger.info(f"Reference: {all_references[i]}")
        logger.info(f"Prediction: {all_predictions[i]}")
        logger.info("---")

    model.train()
    return metrics


if __name__ == "__main__":
    main()