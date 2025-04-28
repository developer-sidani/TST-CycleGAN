# Multi30k Image-Text Translation with TST-CycleGAN

This document provides detailed information about using the Multi30k dataset with TST-CycleGAN for multilingual image-text translation tasks.

## About Multi30k

The Multi30k dataset is a multilingual extension of the Flickr30k dataset, containing image descriptions in English, German, French, and Czech. Each image has a corresponding caption in multiple languages, making it suitable for multilingual image-caption translation tasks.

Key features of Multi30k:
- 29,000 training images with captions
- 1,014 validation images with captions
- Multiple test sets from different years and sources:
  - test_2016_flickr: 1,000 images
  - test_2017_flickr: 1,000 images
  - test_2017_mscoco: 461 images
  - test_2018_flickr: Available for some language pairs

## Dataset Organization

The Multi30k dataset should be organized in the following structure:

```
data/multi30k/
├── data/
│   └── task1/
│       ├── raw/              # Raw text files (.en, .de, .fr, .cs)
│       │   ├── train.en
│       │   ├── train.de
│       │   ├── train.fr
│       │   ├── train.cs
│       │   ├── val.en
│       │   └── ...
│       ├── image_splits/     # Text files with image filenames
│       │   ├── train.txt
│       │   ├── val.txt
│       │   └── test_*_*.txt
│       └── processed/        # TSV files created by scripts (auto-generated)
└── images/
    ├── train/                # Training images
    ├── val/                  # Validation images
    ├── test_2016_flickr/     # Test images
    ├── test_2017_flickr/
    └── test_2017_mscoco/
```

## Training Workflow

The training process is divided into three sequential phases:

### Phase 1: Caption Generation

This phase trains a model to generate captions from images in both source and target languages.

```
Caption Phase:
Image → [CLIP + MLP] → [Prefix Embeddings] → [Text Generator] → Caption in lang1/lang2
```

The caption phase generates a shared representation of the image that can be used for generating text in multiple languages.

### Phase 2: Translation

This phase builds on the caption phase by training a model to translate between languages.

```
Translation Phase:
Text in lang1 → [Text Encoder] → [Translation Model] → Text in lang2
```

The translation phase uses the prefix embeddings from the caption phase as input to generate translations.

### Phase 3: Cycle Consistency

This phase introduces cycle consistency to improve translation quality.

```
Cycle Consistency:
Text in lang1 → [Translation to lang2] → [Translation back to lang1] → Text in lang1 (should match original)
Text in lang2 → [Translation to lang1] → [Translation back to lang2] → Text in lang2 (should match original)
```

The cycle consistency phase ensures that translations preserve the meaning of the original text.

## Available SBATCH Scripts

We provide several SBATCH scripts to facilitate training on HPC clusters:

| Script | Description |
|--------|-------------|
| `train_multi30k_caption.sbatch` | Trains the caption phase (Phase 1) |
| `train_multi30k_translate.sbatch` | Trains the translation phase (Phase 2) |
| `train_multi30k_cycle.sbatch` | Trains the cycle consistency phase (Phase 3) |
| `train_multi30k_all.sbatch` | Runs all three phases in sequence |

### Running the Scripts

Each script accepts parameters for language pairs and test sets:

```bash
sbatch scripts/train_multi30k_all.sbatch [SRC_LANG] [TGT_LANG] [TEST_YEAR] [TEST_SET]
```

Example:
```bash
# Train English to German translation using the 2017 Flickr test set
sbatch scripts/train_multi30k_all.sbatch en de 2017 flickr
```

Default values if not specified:
- SRC_LANG: en
- TGT_LANG: de
- TEST_YEAR: 2016
- TEST_SET: flickr

### Data Preprocessing

The scripts automatically preprocess the raw text files from the Multi30k dataset by:

1. Creating a processed directory if it doesn't exist
2. Generating TSV files that combine image filenames with captions
3. Using the `paste` command to merge image filenames and captions

Example of automatic preprocessing:
```bash
paste -d '\t' $TRAIN_SPLITS "/path/to/train.en" > "processed/train_en.tsv"
```

## Model Architecture

The training uses several key components:

- **Image Encoder**: CLIP (openai/clip-vit-base-patch32)
- **Text Generator**: mBART (facebook/mbart-large-50)
- **Discriminator**: DistilBERT (distilbert-base-multilingual-cased)
- **Mapping Network**: MLP (default) or transformer

## Model Outputs

After training, the models are saved in the following directories:

- Caption Phase: `./models/multi30k_[SRC]_[TGT]_caption_p1/`
- Translate Phase: `./models/multi30k_[SRC]_[TGT]_translate_p2/`
- Cycle Phase: `./models/multi30k_[SRC]_[TGT]_cycle_p3/`

Each directory contains checkpoints at different epochs, with the final model in the `final/` subdirectory.