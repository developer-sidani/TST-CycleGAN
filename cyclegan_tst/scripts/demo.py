"""
Demo script for Multimodal CycleGAN MMT.
Allows translating text with or without images using a trained model.
"""

import os
import argparse
import logging
import sys
import glob
from PIL import Image

import torch
from torch.cuda.amp import autocast

# Import custom modules
from cyclegan_tst.models.MultimodalGeneratorModel import MultimodalGeneratorModel
from cyclegan_tst.models.DiscriminatorModel import DiscriminatorModel
from cyclegan_tst.models.MultimodalCycleGANModel import MultimodalCycleGANModel

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Demo for Multimodal CycleGAN MMT")

    # Model arguments
    parser.add_argument("--model_dir", type=str, required=True,
                        help="Directory containing the trained model")
    parser.add_argument("--src_lang", type=str, default="en",
                        help="Source language")
    parser.add_argument("--tgt_lang", type=str, default="de",
                        help="Target language")
    parser.add_argument("--clip_model", type=str, default="M-CLIP/XLM-Roberta-Large-Vit-B-32",
                        help="CLIP model to use")
    parser.add_argument("--adapter_type", type=str, default="mlp",
                        choices=["mlp", "hidden_mlp", "transformer"],
                        help="Type of adapter to use")
    parser.add_argument("--prefix_length", type=int, default=10,
                        help="Length of prefix for adapter")
    parser.add_argument("--max_seq_length", type=int, default=64,
                        help="Maximum sequence length")

    # Input/output arguments
    parser.add_argument("--image", type=str, default=None,
                        help="Path to input image (optional)")
    parser.add_argument("--input_text", type=str, default=None,
                        help="Input text to translate")
    parser.add_argument("--direction", type=str, default="AB", choices=["AB", "BA"],
                        help="Translation direction: AB (src->tgt) or BA (tgt->src)")
    parser.add_argument("--interactive", action="store_true",
                        help="Run in interactive mode")
    parser.add_argument("--fp16", action="store_true",
                        help="Use mixed precision")

    return parser.parse_args()


def load_model(model_dir, src_lang, tgt_lang, clip_model, adapter_type, prefix_length, max_seq_length, device):
    """Load model from directory"""
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

    # Initialize dummy discriminators (not needed for inference)
    D_a = DiscriminatorModel(
        model_name_or_path="distilbert-base-multilingual-cased",
        max_seq_length=max_seq_length
    )

    D_b = DiscriminatorModel(
        model_name_or_path="distilbert-base-multilingual-cased",
        max_seq_length=max_seq_length
    )

    # Create CycleGAN model
    cycle_gan = MultimodalCycleGANModel(
        G_ab=G_ab,
        G_ba=G_ba,
        D_a=D_a,
        D_b=D_b,
        Cls=None,
        device=device
    )

    return cycle_gan


def translate(cycle_gan, text, image_path=None, direction="AB", device=None, fp16=False):
    """Translate text using the model"""
    # Prepare image if provided
    image = None
    if image_path and os.path.exists(image_path):
        try:
            image = Image.open(image_path).convert("RGB")
            logger.info(f"Loaded image from {image_path}")
        except Exception as e:
            logger.error(f"Error loading image: {e}")
            image = None

    cycle_gan.eval()

    with torch.no_grad():
        with autocast(enabled=fp16):
            # Use transfer method for translation
            if direction == "AB":
                # Source -> Target translation
                translations = cycle_gan.transfer(
                    sentences=[text],
                    direction="AB",
                    images=[image] if image else None,
                    device=device
                )
            else:
                # Target -> Source translation
                translations = cycle_gan.transfer(
                    sentences=[text],
                    direction="BA",
                    images=[image] if image else None,
                    device=device
                )

    # Return the translation
    return translations[0]


def interactive_mode(cycle_gan, src_lang, tgt_lang, device, fp16):
    """Run in interactive mode for continuous translations"""
    print(f"\n===== Multimodal CycleGAN NMT Demo =====")
    print(f"Source language: {src_lang}, Target language: {tgt_lang}")
    print("Enter text to translate, or 'q' to quit.")
    print("You can specify an image path by prefixing with 'image:' (e.g., 'image:my_image.jpg text to translate')")
    print("You can switch direction by typing 'switch'")

    direction = "AB"  # Default: source -> target
    current_src = src_lang
    current_tgt = tgt_lang

    while True:
        # Print current direction
        print(f"\nCurrent direction: {current_src} -> {current_tgt}")

        # Get input
        user_input = input(f"{current_src}> ")

        # Check for exit command
        if user_input.lower() in ('q', 'quit', 'exit'):
            print("Exiting...")
            break

        # Check for direction switch
        if user_input.lower() == 'switch':
            direction = "BA" if direction == "AB" else "AB"
            current_src, current_tgt = current_tgt, current_src
            print(f"Direction switched: {current_src} -> {current_tgt}")
            continue

        # Check for image prefix
        image_path = None
        if user_input.startswith('image:'):
            parts = user_input.split(' ', 1)
            if len(parts) > 1:
                image_path = parts[0][6:]  # Remove 'image:' prefix
                user_input = parts[1]
                print(f"Using image: {image_path}")

        # Translate
        try:
            translation = translate(
                cycle_gan=cycle_gan,
                text=user_input,
                image_path=image_path,
                direction=direction,
                device=device,
                fp16=fp16
            )

            # Print result
            print(f"{current_tgt}> {translation}")
        except Exception as e:
            print(f"Error during translation: {e}")


def main():
    """Main function"""
    # Parse arguments
    args = parse_args()

    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load model
    cycle_gan = load_model(
        model_dir=args.model_dir,
        src_lang=args.src_lang,
        tgt_lang=args.tgt_lang,
        clip_model=args.clip_model,
        adapter_type=args.adapter_type,
        prefix_length=args.prefix_length,
        max_seq_length=args.max_seq_length,
        device=device
    )

    # Run interactive mode if specified
    if args.interactive:
        interactive_mode(
            cycle_gan=cycle_gan,
            src_lang=args.src_lang,
            tgt_lang=args.tgt_lang,
            device=device,
            fp16=args.fp16
        )
    else:
        # Check if input text is provided
        if not args.input_text:
            logger.error("Input text is required for non-interactive mode")
            sys.exit(1)

        # Translate
        translation = translate(
            cycle_gan=cycle_gan,
            text=args.input_text,
            image_path=args.image,
            direction=args.direction,
            device=device,
            fp16=args.fp16
        )

        # Print result
        print(f"Input ({args.src_lang if args.direction == 'AB' else args.tgt_lang}): {args.input_text}")
        print(f"Translation ({args.tgt_lang if args.direction == 'AB' else args.src_lang}): {translation}")


if __name__ == "__main__":
    main()