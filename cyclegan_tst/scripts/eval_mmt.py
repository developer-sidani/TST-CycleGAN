"""
Evaluation script for Multimodal CycleGAN MMT.
Evaluates trained models on various test sets: Flickr 2016, 2017, 2018, and MSCOCO 2017.
"""

import os
import argparse
import logging
import random
import numpy as np
import json
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv

import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
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
    parser = argparse.ArgumentParser(description="Evaluation script for Multimodal CycleGAN MMT")

    # Data arguments
    parser.add_argument("--data_dir", type=str, default="data/multi30k",
                        help="Directory containing Multi30k data")
    parser.add_argument("--image_dir", type=str, default="data/multi30k/images",
                        help="Directory containing Multi30k images")
    parser.add_argument("--cache_dir", type=str, default="data/multi30k/cache",
                        help="Directory to cache processed data")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save results")
    parser.add_argument("--src_lang", type=str, default="en",
                        help="Source language")
    parser.add_argument("--tgt_lang", type=str, default="de",
                        help="Target language")
    parser.add_argument("--model_dir", type=str, required=True,
                        help="Directory containing the trained model")

    # Test set arguments
    parser.add_argument("--test_all", action="store_true",
                        help="Evaluate on all available test sets")
    parser.add_argument("--test_sets", nargs="+", default=["2016 flickr"],
                        help="Test sets to evaluate on, e.g., --test_sets 2016 flickr 2017 mscoco")

    # Model arguments
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

    # Evaluation arguments
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for evaluation")
    parser.add_argument("--include_images", action="store_true",
                        help="Include images in evaluation (for multimodal translation)")
    parser.add_argument("--beam_size", type=int, default=4,
                        help="Beam size for generation")
    parser.add_argument("--fp16", action="store_true",
                        help="Use mixed precision evaluation")
    parser.add_argument("--save_translations", action="store_true",
                        help="Save translations to file")
    parser.add_argument("--compute_bleu", action="store_true", default=True,
                        help="Compute BLEU score")
    parser.add_argument("--compute_meteor", action="store_true", default=True,
                        help="Compute METEOR score")
    parser.add_argument("--compute_bertscore", action="store_true",
                        help="Compute BERTScore")
    parser.add_argument("--compute_comet", action="store_true",
                        help="Compute COMET score")

    # Comparison arguments
    parser.add_argument("--compare_with", nargs="+", default=[],
                        help="Paths to other models to compare with")
    parser.add_argument("--compare_labels", nargs="+", default=[],
                        help="Labels for models being compared")

    # Comet ML arguments
    parser.add_argument("--use_comet", action="store_true",
                        help="Whether to use Comet ML for logging")
    parser.add_argument("--comet_project", type=str, default="multimodal-cyclegan-nmt",
                        help="Comet ML project name")
    parser.add_argument("--comet_workspace", type=str, default=None,
                        help="Comet ML workspace")

    return parser.parse_args()


def load_model(model_dir, src_lang, tgt_lang, clip_model, adapter_type, prefix_length, max_seq_length, disc_model, device):
    """Load CycleGAN model from directory"""
    logger.info(f"Loading model from {model_dir}")

    # Load G_AB (source -> target)
    g_ab_path = os.path.join(model_dir, "G_ab")
    if os.path.exists(g_ab_path):
        logger.info(f"Loading G_AB from {g_ab_path}")
        G_ab = MultimodalGeneratorModel(
            mbart_model_path="facebook/mbart-large-50",  # Will be overridden by pretrained
            clip_model_path=clip_model,
            adapter_type=adapter_type,
            prefix_length=prefix_length,
            max_seq_length=max_seq_length,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            pretrained_path=g_ab_path
        )
    else:
        raise ValueError(f"G_AB not found in {model_dir}")

    # Load G_BA (target -> source)
    g_ba_path = os.path.join(model_dir, "G_ba")
    if os.path.exists(g_ba_path):
        logger.info(f"Loading G_BA from {g_ba_path}")
        G_ba = MultimodalGeneratorModel(
            mbart_model_path="facebook/mbart-large-50",
            clip_model_path=clip_model,
            adapter_type=adapter_type,
            prefix_length=prefix_length,
            max_seq_length=max_seq_length,
            src_lang=tgt_lang,  # Reversed for G_BA
            tgt_lang=src_lang,  # Reversed for G_BA
            pretrained_path=g_ba_path
        )
    else:
        raise ValueError(f"G_BA not found in {model_dir}")

    # Load discriminators (not essential for inference, but needed for model structure)
    d_a_path = os.path.join(model_dir, "D_a")
    if os.path.exists(d_a_path):
        logger.info(f"Loading D_A from {d_a_path}")
        D_a = DiscriminatorModel(
            model_name_or_path=disc_model,
            pretrained_path=d_a_path,
            max_seq_length=max_seq_length
        )
    else:
        logger.warning(f"D_A not found in {model_dir}, initializing new")
        D_a = DiscriminatorModel(
            model_name_or_path=disc_model,
            max_seq_length=max_seq_length
        )

    d_b_path = os.path.join(model_dir, "D_b")
    if os.path.exists(d_b_path):
        logger.info(f"Loading D_B from {d_b_path}")
        D_b = DiscriminatorModel(
            model_name_or_path=disc_model,
            pretrained_path=d_b_path,
            max_seq_length=max_seq_length
        )
    else:
        logger.warning(f"D_B not found in {model_dir}, initializing new")
        D_b = DiscriminatorModel(
            model_name_or_path=disc_model,
            max_seq_length=max_seq_length
        )

    # Initialize CycleGAN model
    cycle_gan = MultimodalCycleGANModel(
        G_ab=G_ab,
        G_ba=G_ba,
        D_a=D_a,
        D_b=D_b,
        Cls=None,  # Classifier not needed for inference
        device=device
    )

    return cycle_gan


def evaluate_model(cycle_gan, test_dataloader, device, include_images=False, fp16=False, beam_size=4):
    """Evaluate model on test set"""
    cycle_gan.eval()

    all_src_texts = []
    all_tgt_texts = []
    all_ab_translations = []  # source -> target
    all_ba_translations = []  # target -> source

    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Translating"):
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

    return all_src_texts, all_tgt_texts, all_ab_translations, all_ba_translations


def compute_metrics(src_texts, tgt_texts, ab_translations, ba_translations, compute_bleu=True, compute_meteor=True,
                  compute_bertscore=False, compute_comet=False, lang="en"):
    """Compute evaluation metrics"""
    metrics = {}

    # Calculate BLEU scores
    if compute_bleu:
        ab_bleu = sacrebleu.corpus_bleu(ab_translations, [tgt_texts]).score
        ba_bleu = sacrebleu.corpus_bleu(ba_translations, [src_texts]).score
        avg_bleu = (ab_bleu + ba_bleu) / 2

        metrics["ab_bleu"] = ab_bleu
        metrics["ba_bleu"] = ba_bleu
        metrics["avg_bleu"] = avg_bleu

    # Calculate METEOR scores
    if compute_meteor:
        meteor = evaluate.load('meteor')
        ab_meteor = meteor.compute(predictions=ab_translations, references=tgt_texts)['meteor'] * 100
        ba_meteor = meteor.compute(predictions=ba_translations, references=src_texts)['meteor'] * 100
        avg_meteor = (ab_meteor + ba_meteor) / 2

        metrics["ab_meteor"] = ab_meteor
        metrics["ba_meteor"] = ba_meteor
        metrics["avg_meteor"] = avg_meteor

    # Calculate BERTScore
    if compute_bertscore:
        try:
            bertscore = evaluate.load('bertscore')
            ab_bertscore = bertscore.compute(predictions=ab_translations, references=tgt_texts, lang=lang)
            ba_bertscore = bertscore.compute(predictions=ba_translations, references=src_texts, lang=lang)

            ab_bertscore_f1 = np.mean(ab_bertscore['f1']) * 100
            ba_bertscore_f1 = np.mean(ba_bertscore['f1']) * 100
            avg_bertscore_f1 = (ab_bertscore_f1 + ba_bertscore_f1) / 2

            metrics["ab_bertscore"] = ab_bertscore_f1
            metrics["ba_bertscore"] = ba_bertscore_f1
            metrics["avg_bertscore"] = avg_bertscore_f1
        except Exception as e:
            logger.warning(f"Error computing BERTScore: {e}")

    # Calculate COMET scores
    if compute_comet:
        try:
            from unbabel.comet import download_model, load_from_checkpoint

            # Download and load COMET model
            comet_path = download_model("wmt20-comet-da")
            comet_model = load_from_checkpoint(comet_path)

            # Prepare data for COMET
            ab_comet_data = [
                {"src": src, "mt": mt, "ref": ref}
                for src, mt, ref in zip(src_texts, ab_translations, tgt_texts)
            ]

            ba_comet_data = [
                {"src": src, "mt": mt, "ref": ref}
                for src, mt, ref in zip(tgt_texts, ba_translations, src_texts)
            ]

            # Compute COMET scores
            ab_comet_scores = comet_model.predict(ab_comet_data, batch_size=8, gpus=1)
            ba_comet_scores = comet_model.predict(ba_comet_data, batch_size=8, gpus=1)

            ab_comet = ab_comet_scores.system_score * 100
            ba_comet = ba_comet_scores.system_score * 100
            avg_comet = (ab_comet + ba_comet) / 2

            metrics["ab_comet"] = ab_comet
            metrics["ba_comet"] = ba_comet
            metrics["avg_comet"] = avg_comet
        except Exception as e:
            logger.warning(f"Error computing COMET score: {e}")

    return metrics


def main():
    """Main evaluation function"""
    # Parse arguments
    args = parse_args()

    # Load environment variables for Comet ML
    if args.use_comet:
        load_dotenv()

    # Setup logging
    os.makedirs(args.output_dir, exist_ok=True)
    log_file = os.path.join(args.output_dir, "evaluation.log")
    file_handler = logging.FileHandler(log_file)
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

    # Parse test sets
    if args.test_all:
        test_sets = [
            ("2016", "flickr"),
            ("2016", "val"),
            ("2017", "flickr"),
            ("2017", "mscoco"),
            ("2018", "flickr")
        ]
    else:
        test_sets = []
        for i in range(0, len(args.test_sets), 2):
            if i + 1 < len(args.test_sets):
                test_sets.append((args.test_sets[i], args.test_sets[i+1]))

    logger.info(f"Evaluating on test sets: {test_sets}")

    # Initialize data loader
    logger.info("Loading Multi30k data...")
    data_loader = Multi30kDataLoader(
        data_dir=args.data_dir,
        langs=[args.src_lang, args.tgt_lang],
        test_sets=test_sets,
        image_dir=args.image_dir,
        cache_dir=args.cache_dir
    )

    # Load model
    model_dirs = [args.model_dir] + args.compare_with
    model_labels = ["Our Model"] + args.compare_labels

    if len(model_labels) < len(model_dirs):
        # Generate default labels if not provided
        model_labels = [f"Model {i+1}" for i in range(len(model_dirs))]

    # Store results for all models
    all_results = {}

    # Evaluate each model on each test set
    for model_idx, model_dir in enumerate(model_dirs):
        model_label = model_labels[model_idx]
        logger.info(f"Evaluating {model_label} from {model_dir}")

        # Load model
        cycle_gan = load_model(
            model_dir=model_dir,
            src_lang=args.src_lang,
            tgt_lang=args.tgt_lang,
            clip_model=args.clip_model,
            adapter_type=args.adapter_type,
            prefix_length=args.prefix_length,
            max_seq_length=args.max_seq_length,
            disc_model=args.disc_model,
            device=device
        )

        # Store results for this model
        model_results = {}

        # Evaluate on each test set
        for test_set in test_sets:
            test_name = f"{test_set[0]}_{test_set[1]}"
            logger.info(f"Evaluating on {test_name}")

            # Load test data
            test_src, test_tgt, test_images = data_loader.get_test_data(
                test_set=test_set,
                src_lang=args.src_lang,
                tgt_lang=args.tgt_lang,
                include_images=args.include_images
            )

            # Create test dataset and dataloader
            test_dataset = MultimodalDataset(test_src, test_tgt, test_images)
            test_dataloader = DataLoader(
                test_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=4,
                pin_memory=True,
                collate_fn=multimodal_collate_fn
            )

            # Evaluate model
            src_texts, tgt_texts, ab_translations, ba_translations = evaluate_model(
                cycle_gan=cycle_gan,
                test_dataloader=test_dataloader,
                device=device,
                include_images=args.include_images,
                fp16=args.fp16,
                beam_size=args.beam_size
            )

            # Compute metrics
            metrics = compute_metrics(
                src_texts=src_texts,
                tgt_texts=tgt_texts,
                ab_translations=ab_translations,
                ba_translations=ba_translations,
                compute_bleu=args.compute_bleu,
                compute_meteor=args.compute_meteor,
                compute_bertscore=args.compute_bertscore,
                compute_comet=args.compute_comet,
                lang=args.src_lang
            )

            # Save translations if requested
            if args.save_translations:
                output_dir = os.path.join(args.output_dir, model_label, test_name)
                os.makedirs(output_dir, exist_ok=True)

                # Save A->B translations
                with open(os.path.join(output_dir, "ab_translations.txt"), 'w') as f:
                    for src, hyp, ref in zip(src_texts, ab_translations, tgt_texts):
                        f.write(f"Source: {src}\n")
                        f.write(f"Translation: {hyp}\n")
                        f.write(f"Reference: {ref}\n")
                        f.write("\n")

                # Save B->A translations
                with open(os.path.join(output_dir, "ba_translations.txt"), 'w') as f:
                    for src, hyp, ref in zip(tgt_texts, ba_translations, src_texts):
                        f.write(f"Source: {src}\n")
                        f.write(f"Translation: {hyp}\n")
                        f.write(f"Reference: {ref}\n")
                        f.write("\n")

                # Save just the translations (for further evaluation)
                with open(os.path.join(output_dir, "ab_output.txt"), 'w') as f:
                    f.write('\n'.join(ab_translations))

                with open(os.path.join(output_dir, "ba_output.txt"), 'w') as f:
                    f.write('\n'.join(ba_translations))

            # Log metrics
            logger.info(f"Results for {model_label} on {test_name}:")
            for k, v in metrics.items():
                logger.info(f"  {k}: {v:.2f}")

            # Log to Comet ML
            if comet_experiment is not None:
                for k, v in metrics.items():
                    comet_experiment.log_metric(f"{model_label}_{test_name}_{k}", v)

            # Store metrics
            model_results[test_name] = metrics

        # Save model results
        all_results[model_label] = model_results

    # Save all results as JSON
    results_file = os.path.join(args.output_dir, "all_results.json")
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    # Create summary table
    summary = []
    for model_label, model_results in all_results.items():
        for test_name, metrics in model_results.items():
            result_row = {
                "Model": model_label,
                "Test Set": test_name
            }
            result_row.update(metrics)
            summary.append(result_row)

    # Convert to DataFrame and save as CSV
    if summary:
        df = pd.DataFrame(summary)
        df.to_csv(os.path.join(args.output_dir, "results_summary.csv"), index=False)

        # Print summary table
        logger.info("\nResults Summary:")
        logger.info(df.to_string())

        # Log summary table to Comet
        if comet_experiment is not None:
            comet_experiment.log_table("results_summary.csv", tabular_data=df)

    # End Comet experiment
    if comet_experiment is not None:
        comet_experiment.end()

    logger.info("Evaluation completed!")


if __name__ == "__main__":
    main()