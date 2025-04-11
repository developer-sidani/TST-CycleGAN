"""
Full training script for Multimodal CycleGAN MMT.
Trains the complete CycleGAN model with cycle consistency and adversarial losses.
Assumes pretraining (Stage 1) has already been done.
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
from cyclegan_tst.models.DiscriminatorModel import DiscriminatorModel
from cyclegan_tst.models.ClassifierModel import ClassifierModel
from cyclegan_tst.models.MultimodalCycleGANModel import MultimodalCycleGANModel
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
    parser = argparse.ArgumentParser(description="Full training script for Multimodal CycleGAN MMT")

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
    parser.add_argument("--test_sets", nargs="+", default=[("2016", "flickr")],
                        help="Test sets to evaluate on, e.g., --test_sets 2016 flickr 2017 mscoco")

    # Model arguments
    parser.add_argument("--pretrained_g_ab", type=str, required=True,
                        help="Path to pretrained G_AB model from Stage 1")
    parser.add_argument("--pretrained_g_ba", type=str, default=None,
                        help="Path to pretrained G_BA model (if available)")
    parser.add_argument("--clip_model", type=str, default="M-CLIP/XLM-Roberta-Large-Vit-B-32",
                        help="CLIP model to use")
    parser.add_argument("--adapter_type", type=str, default="mlp",
                        choices=["mlp", "hidden_mlp", "transformer"],
                        help="Type of adapter to use")
    parser.add_argument("--disc_model", type=str, default="distilbert-base-multilingual-cased",
                        help="Discriminator base model")
    parser.add_argument("--prefix_length", type=int, default=10,
                        help="Length of prefix for adapter")
    parser.add_argument("--max_seq_length", type=int, default=64,
                        help="Maximum sequence length")
    parser.add_argument("--use_classifier", action="store_true",
                        help="Whether to use a language classifier for guided training")
    parser.add_argument("--classifier_path", type=str, default=None,
                        help="Path to pretrained language classifier (if use_classifier is True)")

    # Training arguments
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of updates steps to accumulate before backward pass")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
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
    parser.add_argument("--include_images", action="store_true",
                        help="Include images in training (for multimodal translation)")

    # Loss weighting arguments
    parser.add_argument("--lambda_cycle", type=float, default=10.0,
                        help="Weight for cycle consistency loss")
    parser.add_argument("--lambda_gen", type=float, default=1.0,
                        help="Weight for generator adversarial loss")
    parser.add_argument("--lambda_disc_fake", type=float, default=1.0,
                        help="Weight for discriminator loss on fake samples")
    parser.add_argument("--lambda_disc_real", type=float, default=1.0,
                        help="Weight for discriminator loss on real samples")
    parser.add_argument("--lambda_cls", type=float, default=0.0,
                        help="Weight for classifier-guided loss")
    parser.add_argument("--lambda_visual", type=float, default=0.0,
                        help="Weight for visual consistency loss")

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

    # Parse test sets
    test_sets = []
    for i in range(0, len(args.test_sets), 2):
        test_sets.append((args.test_sets[i], args.test_sets[i+1]))

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
        logger.info(f"Comet.ml tracking initiated. View experiment at {comet_experiment.url}")
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
        test_sets=test_sets,
        image_dir=args.image_dir,
        cache_dir=args.cache_dir,
        max_samples=args.max_samples
    )

    # Get train and test data
    logger.info("Creating datasets...")
    train_src, train_tgt, train_images = data_loader.get_train_data(
        src_lang=args.src_lang,
        tgt_lang=args.tgt_lang,
        include_images=args.include_images
    )

    # Create evaluation dictionaries for each test set
    test_dataloaders = {}
    for test_set in test_sets:
        test_src, test_tgt, test_images = data_loader.get_test_data(
            test_set=test_set,
            src_lang=args.src_lang,
            tgt_lang=args.tgt_lang,
            include_images=args.include_images
        )

        test_dataset = MultimodalDataset(test_src, test_tgt, test_images)
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            collate_fn=multimodal_collate_fn
        )

        test_dataloaders[f"{test_set[0]}_{test_set[1]}"] = test_dataloader

    # Create training dataset and dataloader
    train_dataset = MultimodalDataset(train_src, train_tgt, train_images)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=multimodal_collate_fn
    )

    # Initialize models
    logger.info("Initializing models...")

    # Load/initialize G_AB (source -> target generator)
    G_ab = MultimodalGeneratorModel(
        mbart_model_path="facebook/mbart-large-50",  # Will be overridden by pretrained weights
        clip_model_path=args.clip_model,
        adapter_type=args.adapter_type,
        prefix_length=args.prefix_length,
        max_seq_length=args.max_seq_length,
        src_lang=args.src_lang,
        tgt_lang=args.tgt_lang,
        pretrained_path=args.pretrained_g_ab
    )

    # Load/initialize G_BA (target -> source generator)
    if args.pretrained_g_ba:
        # If we have a pretrained G_BA, load it
        G_ba = MultimodalGeneratorModel(
            mbart_model_path="facebook/mbart-large-50",
            clip_model_path=args.clip_model,
            adapter_type=args.adapter_type,
            prefix_length=args.prefix_length,
            max_seq_length=args.max_seq_length,
            src_lang=args.tgt_lang,  # Reversed for G_BA
            tgt_lang=args.src_lang,  # Reversed for G_BA
            pretrained_path=args.pretrained_g_ba
        )
    else:
        # If no pretrained G_BA, initialize a new one
        G_ba = MultimodalGeneratorModel(
            mbart_model_path="facebook/mbart-large-50",
            clip_model_path=args.clip_model,
            adapter_type=args.adapter_type,
            prefix_length=args.prefix_length,
            max_seq_length=args.max_seq_length,
            src_lang=args.tgt_lang,  # Reversed for G_BA
            tgt_lang=args.src_lang,  # Reversed for G_BA
        )

    # Initialize discriminators
    D_a = DiscriminatorModel(
        model_name_or_path=args.disc_model,
        max_seq_length=args.max_seq_length
    )

    D_b = DiscriminatorModel(
        model_name_or_path=args.disc_model,
        max_seq_length=args.max_seq_length
    )

    # Initialize classifier if needed
    Cls = None
    if args.use_classifier and args.classifier_path:
        Cls = ClassifierModel(
            pretrained_path=args.classifier_path,
            max_seq_length=args.max_seq_length
        )

    # Initialize CycleGAN model
    cycle_gan = MultimodalCycleGANModel(
        G_ab=G_ab,
        G_ba=G_ba,
        D_a=D_a,
        D_b=D_b,
        Cls=Cls,
        device=device
    )

    # Setup optimizer and scheduler
    optimizer = AdamW(
        cycle_gan.get_optimizer_parameters(),
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

    # Resume from checkpoint if provided
    global_step = 0
    if args.resume_from_checkpoint:
        logger.info(f"Loading checkpoint from {args.resume_from_checkpoint}")
        checkpoint = torch.load(os.path.join(args.resume_from_checkpoint, "checkpoint.pt"), map_location=device)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        global_step = checkpoint["global_step"]

        # Also load models
        cycle_gan.G_ab.load_state_dict(checkpoint["G_ab"])
        cycle_gan.G_ba.load_state_dict(checkpoint["G_ba"])
        cycle_gan.D_a.load_state_dict(checkpoint["D_a"])
        cycle_gan.D_b.load_state_dict(checkpoint["D_b"])

    # Initialize loss tracking
    loss_logging = {
        "Cycle Loss A-B-A": [],
        "Cycle Loss B-A-B": [],
        "Loss generator A-B": [],
        "Loss generator B-A": [],
        "Loss D_A": [],
        "Loss D_B": [],
        "Direct Loss A-B": [],
        "Direct Loss B-A": []
    }

    if args.use_classifier:
        loss_logging["Classifier-guided A-B"] = []
        loss_logging["Classifier-guided B-A"] = []

    if args.include_images and args.lambda_visual > 0:
        loss_logging["Visual Loss A-B"] = []

    # Training loop
    logger.info("Starting training...")

    for epoch in range(args.num_epochs):
        cycle_gan.train()
        epoch_loss = 0

        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{args.num_epochs}")

        for step, batch in enumerate(progress_bar):
            # Get batch data
            src_texts = batch["src_texts"]
            tgt_texts = batch["tgt_texts"]
            images = batch["images"].to(device) if "images" in batch else None

            # Set up lambdas for loss weighting
            lambdas = [
                args.lambda_cycle,
                args.lambda_gen,
                args.lambda_disc_fake,
                args.lambda_disc_real,
                args.lambda_cls,
                args.lambda_visual
            ]

            # Forward pass and compute losses with mixed precision
            with autocast(enabled=args.fp16):
                # Train with multimodal cycle
                if args.include_images and images is not None:
                    cycle_gan.training_cycle_multimodal(
                        src_sentences=src_texts,
                        tgt_sentences=tgt_texts,
                        images=images,
                        lambdas=lambdas,
                        comet_experiment=comet_experiment,
                        loss_logging=loss_logging,
                        training_step=global_step
                    )
                else:
                    # Train with text-only cycle
                    cycle_gan.training_cycle_multimodal(
                        src_sentences=src_texts,
                        tgt_sentences=tgt_texts,
                        images=None,
                        lambdas=lambdas,
                        comet_experiment=comet_experiment,
                        loss_logging=loss_logging,
                        training_step=global_step
                    )

            # Update parameters
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(cycle_gan.get_optimizer_parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(cycle_gan.get_optimizer_parameters(), 1.0)
                    optimizer.step()

                scheduler.step()
                optimizer.zero_grad()

                # Logging for current step
                if global_step % args.logging_steps == 0:
                    # Log losses to progress bar
                    progress_msg = {}
                    for key in loss_logging:
                        if loss_logging[key]:
                            progress_msg[key] = np.mean(loss_logging[key][-100:])

                    progress_bar.set_postfix(progress_msg)

                # Save checkpoint
                if global_step % args.save_steps == 0:
                    save_checkpoint(
                        cycle_gan=cycle_gan,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        global_step=global_step,
                        output_dir=os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    )

                # Evaluate model
                if global_step % args.eval_steps == 0:
                    for test_name, test_dataloader in test_dataloaders.items():
                        eval_metrics = evaluate_model(
                            cycle_gan=cycle_gan,
                            test_dataloader=test_dataloader,
                            device=device,
                            fp16=args.fp16,
                            include_images=args.include_images
                        )

                        logger.info(f"Eval on {test_name} at step {global_step}: {eval_metrics}")

                        if comet_experiment is not None:
                            for metric_name, metric_value in eval_metrics.items():
                                comet_experiment.log_metric(
                                    f"eval_{test_name}_{metric_name}",
                                    metric_value,
                                    step=global_step
                                )

                global_step += 1

        # End of epoch - evaluate on all test sets
        for test_name, test_dataloader in test_dataloaders.items():
            eval_metrics = evaluate_model(
                cycle_gan=cycle_gan,
                test_dataloader=test_dataloader,
                device=device,
                fp16=args.fp16,
                include_images=args.include_images
            )

            logger.info(f"Epoch {epoch+1}/{args.num_epochs} - Eval on {test_name}: {eval_metrics}")

            if comet_experiment is not None:
                for metric_name, metric_value in eval_metrics.items():
                    comet_experiment.log_metric(
                        f"epoch_{test_name}_{metric_name}",
                        metric_value,
                        step=epoch
                    )

        # Save model at end of epoch
        save_checkpoint(
            cycle_gan=cycle_gan,
            optimizer=optimizer,
            scheduler=scheduler,
            global_step=global_step,
            output_dir=os.path.join(args.output_dir, f"epoch-{epoch+1}")
        )

    # Save final model
    save_checkpoint(
        cycle_gan=cycle_gan,
        optimizer=optimizer,
        scheduler=scheduler,
        global_step=global_step,
        output_dir=os.path.join(args.output_dir, "final_model")
    )

    # End Comet experiment
    if comet_experiment is not None:
        comet_experiment.end()

    logger.info("Training completed!")


def save_checkpoint(cycle_gan, optimizer, scheduler, global_step, output_dir):
    """Save model checkpoint"""
    os.makedirs(output_dir, exist_ok=True)

    # Save models
    cycle_gan.save_models(output_dir)

    # Save optimizer, scheduler and training state
    checkpoint = {
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "global_step": global_step,
        "G_ab": cycle_gan.G_ab.state_dict(),
        "G_ba": cycle_gan.G_ba.state_dict(),
        "D_a": cycle_gan.D_a.state_dict(),
        "D_b": cycle_gan.D_b.state_dict()
    }

    torch.save(checkpoint, os.path.join(output_dir, "checkpoint.pt"))
    logger.info(f"Saved checkpoint to {output_dir}")


def evaluate_model(cycle_gan, test_dataloader, device, fp16=False, include_images=False):
    """Evaluate model on test set"""
    cycle_gan.eval()

    all_src_texts = []
    all_tgt_texts = []
    all_ab_translations = []  # source -> target
    all_ba_translations = []  # target -> source

    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Evaluating"):
            # Get batch data
            src_texts = batch["src_texts"]
            tgt_texts = batch["tgt_texts"]
            images = batch["images"].to(device) if "images" in batch and include_images else None

            # Translate in both directions
            with autocast(enabled=fp16):
                # A -> B (source -> target)
                ab_translations = cycle_gan.transfer(
                    sentences=src_texts,
                    direction="AB",
                    images=images,
                    device=device
                )

                # B -> A (target -> source)
                ba_translations = cycle_gan.transfer(
                    sentences=tgt_texts,
                    direction="BA",
                    images=images,
                    device=device
                )

            # Collect results
            all_src_texts.extend(src_texts)
            all_tgt_texts.extend(tgt_texts)
            all_ab_translations.extend(ab_translations)
            all_ba_translations.extend(ba_translations)

    # Calculate metrics for A -> B
    ab_bleu = sacrebleu.corpus_bleu(all_ab_translations, [all_tgt_texts]).score

    meteor = evaluate.load('meteor')
    ab_meteor = meteor.compute(predictions=all_ab_translations, references=all_tgt_texts)['meteor'] * 100

    # Calculate metrics for B -> A
    ba_bleu = sacrebleu.corpus_bleu(all_ba_translations, [all_src_texts]).score
    ba_meteor = meteor.compute(predictions=all_ba_translations, references=all_src_texts)['meteor'] * 100

    # Calculate average metrics
    avg_bleu = (ab_bleu + ba_bleu) / 2
    avg_meteor = (ab_meteor + ba_meteor) / 2

    metrics = {
        "ab_bleu": ab_bleu,
        "ab_meteor": ab_meteor,
        "ba_bleu": ba_bleu,
        "ba_meteor": ba_meteor,
        "avg_bleu": avg_bleu,
        "avg_meteor": avg_meteor
    }

    # Log sample translations
    num_samples = min(5, len(all_ab_translations))
    for i in range(num_samples):
        logger.info(f"Source: {all_src_texts[i]}")
        logger.info(f"A->B Translation: {all_ab_translations[i]}")
        logger.info(f"Reference B: {all_tgt_texts[i]}")
        logger.info(f"B->A Translation: {all_ba_translations[i]}")
        logger.info("---")

    cycle_gan.train()
    return metrics


if __name__ == "__main__":
    main()