import csv
from typing import List
import logging
import random
import os
from PIL import Image
import torch

import pandas as pd
import numpy as np

from torch.utils.data import Dataset

class MonostyleDataset(Dataset):
    """
    Mono-style dataset
    """

    def __init__(
        self, 
        dataset_format: str,
        dataset_path: str = None,
        sentences_list: List[str] = None,
        text_column_name: str = None,
        separator: str = None,
        style: str = None,
        max_dataset_samples: int = None,
        ):
        super(MonostyleDataset, self).__init__()

        self.allowed_dataset_formats = ["list", "csv", "line_file"]

        if dataset_format not in self.allowed_dataset_formats:
            raise Exception(f"MonostyleDataset: provided dataset format {dataset_format} is not supported.")

        self.dataset_format = dataset_format
        
        # Checking for list
        if self.dataset_format == "list" and sentences_list is None:
            raise Exception(f"MonostyleDataset: provided dataset format {dataset_format} but required argument 'sentences_list' is not provided.")
        elif self.dataset_format == "list":
            self.data = self.sentences_list

        # Checking for csv
        if self.dataset_format in ["csv"] and dataset_path is None:
            raise Exception(f"MonostyleDataset: provided dataset format {dataset_format} but required argument 'dataset_path' is not provided.")
        elif self.dataset_format in ["csv"] and text_column_name is None:
            raise Exception(f"MonostyleDataset: provided dataset format {dataset_format} but required argument 'text_column_name' is not provided.")
        elif self.dataset_format in ["csv"] and separator is None:
            raise Exception(f"MonostyleDataset: provided dataset format {dataset_format} but required argument 'separator' is not provided.")
        
        # Checking for line_file
        if self.dataset_format in ["line_file"] and dataset_path is None:
            raise Exception(f"MonostyleDataset: provided dataset format {dataset_format} but required argument 'dataset_path' is not provided.")
        elif self.dataset_format in ["line_file"] and separator is None:
            raise Exception(f"MonostyleDataset: provided dataset format {dataset_format} but required argument 'separator' is not provided.")

        self.dataset_format = dataset_format
        self.dataset_path = dataset_path
        self.text_column_name = text_column_name
        self.separator = separator
        self.style = style
        self.max_dataset_samples = max_dataset_samples

        self.load_data()

    def _load_data_csv(self):
        df = pd.read_csv(self.dataset_path, sep=self.separator, header=None)
        df.dropna(inplace=True)
        self.data = df[self.text_column_name].tolist()
        logging.debug(f"MonostyleDataset, load_data: parsed {len(self.data)} examples")
    
    def _load_data_line_file(self):
        with open(self.dataset_path) as input_file:
            self.data = input_file.read()
            self.data = self.data.split(self.separator)
        logging.debug(f"MonostyleDataset, load_data: parsed {len(self.data)} examples")   

    def load_data(self, SEED=42):

        if self.dataset_format == "csv":
            self._load_data_csv()
        elif self.dataset_format == "line_file":
            self._load_data_line_file()
        elif self.dataset_format == "list":
            logging.debug(f"MonostyleDataset, load_data: data already loaded, {len(self.data)} examples")
        else:
            raise Exception(f"MonostyleDataset, load_data: the {self.dataset_format} is not supported. Please use one of the following {self.allowed_dataset_formats}")

        random.seed(SEED)
        if self.max_dataset_samples is not None and self.max_dataset_samples < len(self.data):
            ix = random.sample(range(0, len(self.data)), self.max_dataset_samples)
            self.data = np.array(self.data)[ix].tolist()
        random.shuffle(self.data)
    
    def reduce_data(self, n_samples):
        self.data = self.data[:n_samples]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class ParallelRefDataset(Dataset):
    """
    ParallelRefDataset dataset
    """

    def __init__(
        self, 
        dataset_format: str,
        dataset_path_src: str = None,
        dataset_path_ref: str = None,
        n_ref: int = None,
        sentences_list_src: List[str] = None,
        sentences_list_ref: List[str] = None,
        text_column_name_src: str = None,
        text_column_name_ref: str = None,
        separator_src: str = None,
        separator_ref: str = None,
        style_src: str = None,
        style_ref: str = None,
        max_dataset_samples: int = None,
        ):
        super(ParallelRefDataset, self).__init__()

        self.allowed_dataset_formats = ["list", "csv", "line_file"]

        if dataset_format not in self.allowed_dataset_formats:
            raise Exception(f"ParallelRefDataset: provided dataset format {dataset_format} is not supported.")

        self.dataset_format = dataset_format
        
        # Checking for list
        if self.dataset_format == "list" and (sentences_list_src is None or sentences_list_ref is None):
            raise Exception(f"ParallelRefDataset: provided dataset format {dataset_format} but required arguments 'sentences_list_a' or 'sentences_list_b' is not provided.")
        elif self.dataset_format == "list":
            self.data_src = sentences_list_src
            self.data_ref = sentences_list_ref
            assert len(self.data_src) == len(self.data_ref)
        
        # Checking for csv
        if self.dataset_format in ["csv"] and (dataset_path_src is None or dataset_path_ref is None):
            raise Exception(f"ParallelRefDataset: provided dataset format {dataset_format} but required argument 'dataset_path_a' or 'dataset_path_b' is not provided.")
        elif self.dataset_format in ["csv"] and (text_column_name_src is None or text_column_name_ref is None):
            raise Exception(f"ParallelRefDataset: provided dataset format {dataset_format} but required argument 'text_column_name_a' or 'text_column_name_b' is not provided.")
        elif self.dataset_format in ["csv"] and (separator_src is None or separator_ref is None):
            raise Exception(f"ParallelRefDataset: provided dataset format {dataset_format} but required argument 'separator_a' or 'separator_b' is not provided.")
        
        # Checking for line_file
        if self.dataset_format in ["line_file"] and (dataset_path_src is None or dataset_path_ref is None):
            raise Exception(f"ParallelRefDataset: provided dataset format {dataset_format} but required argument 'dataset_path_a' or 'dataset_path_b' is not provided.")
        elif self.dataset_format in ["line_file"] and (separator_src is None or separator_ref is None):
            raise Exception(f"ParallelRefDataset: provided dataset format {dataset_format} but required argument 'separator_a' or 'separator_b' is not provided.")

        self.dataset_format = dataset_format
        self.n_ref = n_ref
        self.dataset_path_src = dataset_path_src
        self.dataset_path_ref = dataset_path_ref
        self.text_column_name_src = text_column_name_src
        self.text_column_name_ref = text_column_name_ref
        self.separator_src = separator_src
        self.separator_ref = separator_ref
        self.style_src = style_src
        self.style_ref = style_ref
        self.max_dataset_samples = max_dataset_samples

        self.load_data()

    def _load_data_csv(self):
        df_src = pd.read_csv(self.dataset_path_src, sep=self.separator_src)
        df_ref = pd.read_csv(self.dataset_path_ref, sep=self.separator_ref)
        df_src.dropna(inplace=True)
        df_ref.dropna(inplace=True)
        self.data_src = df_src[self.text_column_name_src].tolist()
        column_names = [self.text_column_name_ref+str(i) for i in range(self.n_ref)]
        self.data_ref = df_ref[column_names].to_numpy().tolist()
        assert len(self.data_src) == len(self.data_ref)
        logging.debug(f"ParallelRefDataset, load_data: parsed {len(self.data_src)} examples")

    def _load_data_line_file(self):
        with open(self.dataset_path_src) as input_file:
            self.data_src = input_file.read()
            self.data_src = self.data_src.split(self.separator_src)
        if self.style_src in ['en']: ref_tag = '0'
        elif self.style_src in ['de', 'fr', 'cs']: ref_tag = '1'
        self.data_ref = [[] for _ in range(len(self.data_src))]
        for i in range(self.n_ref):
            ref_path = self.dataset_path_ref+f'reference{i}.{ref_tag}.txt'
            with open(ref_path) as input_file:        
                cur_ref = input_file.read()
                cur_ref = cur_ref.split(self.separator_ref)
            for i, ref in enumerate(cur_ref):
                self.data_ref[i].append(ref)
        assert len(self.data_src) == len(self.data_ref)
        logging.debug(f"ParallelRefDataset, load_data: parsed {len(self.data_src)} examples")

    def load_data(self, SEED=42):

        if self.dataset_format == "csv":
            self._load_data_csv()
        elif self.dataset_format == "line_file":
            self._load_data_line_file()
        elif self.dataset_format == "list":
            logging.debug(f"ParallelRefDataset, load_data: data already loaded, {len(self.data_src)} examples")
        else:
            raise Exception(f"ParallelRefDataset, load_data: the {self.dataset_format} is not supported. Please use one of the following {self.allowed_dataset_formats}")
        
        if self.max_dataset_samples is not None and self.max_dataset_samples < len(self.data_src):
            random.seed(SEED)
            ix = random.sample(range(0, len(self.data_src)), self.max_dataset_samples)
            self.data_src = np.array(self.data_src)[ix].tolist()
            self.data_ref = np.array(self.data_ref)[ix].tolist()

    def __len__(self):
        return len(self.data_src)

    def __getitem__(self, idx):
        return self.data_src[idx], self.data_ref[idx]

    @staticmethod
    def customCollate(batch):
        src_list = []
        ref_list = []
        for t in batch:
            src_list.append(t[0])
            ref_list.append(t[1])
        src_list = tuple(src_list)
        return [src_list, ref_list]

class ImageCaptionDataset(Dataset):
    """
    Dataset for loading image-caption pairs for pretraining.
    """
    def __init__(
        self, 
        image_dir: str, 
        caption_file: str, 
        transform=None, 
        max_dataset_samples: int = None,
        lang: str = "en"
    ):
        """
        Args:
            image_dir (str): Directory with all the images.
            caption_file (str): Path to the caption file.
            transform (callable, optional): Optional transform to be applied on an image.
            max_dataset_samples (int, optional): Maximum number of samples to use.
            lang (str, optional): Language of captions.
        """
        super(ImageCaptionDataset, self).__init__()
        
        self.image_dir = image_dir
        self.transform = transform
        self.lang = lang
        self.use_placeholder_images = False
        
        print(f"\nDEBUG: Loading ImageCaptionDataset")
        print(f"DEBUG: Caption file: {caption_file}")
        print(f"DEBUG: Image directory: {image_dir}")
        print(f"DEBUG: Language: {lang}")
        
        # Check if image directory exists
        if not os.path.exists(image_dir):
            print(f"WARNING: Image directory does not exist: {image_dir}")
            # Try to find a close match
            parent_dir = os.path.dirname(image_dir)
            if os.path.exists(parent_dir):
                print(f"Parent directory exists: {parent_dir}")
                subdirs = [d for d in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, d))]
                print(f"Available subdirectories: {subdirs}")
                
                # Check for images folder
                if "images" in subdirs:
                    candidate = os.path.join(parent_dir, "images")
                    print(f"Found 'images' directory: {candidate}")
                    self.image_dir = candidate
                # Try other common names
                elif "imgs" in subdirs:
                    candidate = os.path.join(parent_dir, "imgs")
                    print(f"Found 'imgs' directory: {candidate}")
                    self.image_dir = candidate
                elif "flickr30k" in subdirs:
                    candidate = os.path.join(parent_dir, "flickr30k")
                    print(f"Found 'flickr30k' directory: {candidate}")
                    self.image_dir = candidate
                else:
                    print(f"WARNING: No image directory found, switching to placeholder images mode")
                    self.use_placeholder_images = True
            else:
                print(f"WARNING: Parent directory does not exist, switching to placeholder images mode")
                self.use_placeholder_images = True
        else:
            # Check image directory contents
            print(f"DEBUG: Image directory exists, checking contents...")
            img_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
            if not img_files:
                print(f"WARNING: No image files found in {image_dir}, switching to placeholder images mode")
                self.use_placeholder_images = True
            else:
                print(f"DEBUG: Found {len(img_files)} image files. Examples: {img_files[:3]}")
        
        # Load captions
        self.data = []
        
        if caption_file.endswith('.tsv'):
            print(f"DEBUG: Loading TSV caption file")
            # Tab-separated format (image_path \t caption)
            with open(caption_file, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if '\t' in line:
                        img_path, caption = line.strip().split('\t', 1)
                        self.data.append((img_path, caption))
                        # Print first few examples
                        if i < 3:
                            print(f"DEBUG: Sample {i}: Image={img_path}, Caption={caption[:50]}...")
            print(f"DEBUG: Loaded {len(self.data)} image-caption pairs from TSV")
        elif caption_file.endswith('.txt'):
            print(f"DEBUG: Loading TXT caption file")
            # Text file format (one caption per line, first token is image filename)
            with open(caption_file, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    parts = line.strip().split(' ', 1)
                    if len(parts) == 2:
                        img_path, caption = parts
                        self.data.append((img_path, caption))
                        # Print first few examples
                        if i < 3:
                            print(f"DEBUG: Sample {i}: Image={img_path}, Caption={caption[:50]}...")
            print(f"DEBUG: Loaded {len(self.data)} image-caption pairs from TXT")
        elif os.path.basename(caption_file).startswith('train.') or os.path.basename(caption_file).startswith('test_') or os.path.basename(caption_file).startswith('val_'):
            print(f"DEBUG: Loading Multi30k plain text caption file")
            # Multi30k format - one caption per line, no image reference
            # We'll use line numbers to match with image files
            with open(caption_file, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    # Assume image files are named sequentially as COCO_train_000000000001.jpg, etc.
                    # or use i+1 padded to 12 digits for image filename
                    img_name = f"COCO_train_{(i+1):012d}.jpg"
                    caption = line.strip()
                    self.data.append((img_name, caption))
                    # Print first few examples
                    if i < 3:
                        print(f"DEBUG: Sample {i}: Image={img_name}, Caption={caption[:50]}...")
            print(f"DEBUG: Loaded {len(self.data)} image-caption pairs from Multi30k plain text")
        else:
            try:
                print(f"DEBUG: Loading CSV caption file")
                # CSV format
                df = pd.read_csv(caption_file)
                print(f"DEBUG: CSV columns: {df.columns.tolist()}")
                if 'image' in df.columns and 'caption' in df.columns:
                    for i, (_, row) in enumerate(df.iterrows()):
                        self.data.append((row['image'], row['caption']))
                        # Print first few examples
                        if i < 3:
                            print(f"DEBUG: Sample {i}: Image={row['image']}, Caption={row['caption'][:50]}...")
                else:
                    raise ValueError(f"CSV file must have 'image' and 'caption' columns")
                print(f"DEBUG: Loaded {len(self.data)} image-caption pairs from CSV")
            except Exception as e:
                print(f"DEBUG: Error loading CSV: {e}")
                print(f"DEBUG: Falling back to plain text format")
                # Try plain text as a last resort
                with open(caption_file, 'r', encoding='utf-8') as f:
                    for i, line in enumerate(f):
                        # Assume image files are named sequentially
                        img_name = f"COCO_train_{(i+1):012d}.jpg"
                        caption = line.strip()
                        self.data.append((img_name, caption))
                        # Print first few examples
                        if i < 3:
                            print(f"DEBUG: Sample {i}: Image={img_name}, Caption={caption[:50]}...")
                print(f"DEBUG: Loaded {len(self.data)} image-caption pairs from plain text")
        
        # Apply max samples limit
        if max_dataset_samples is not None and max_dataset_samples < len(self.data):
            random.seed(42)  # For reproducibility
            ix = random.sample(range(0, len(self.data)), max_dataset_samples)
            self.data = [self.data[i] for i in ix]
            print(f"DEBUG: Limited dataset to {len(self.data)} samples")
            
        # Verify image paths
        for i, (img_path, _) in enumerate(self.data[:5]):
            full_path = os.path.join(self.image_dir, img_path)
            exists = os.path.exists(full_path)
            print(f"DEBUG: Image {i} path check: {full_path} - Exists: {exists}")
            
        print(f"DEBUG: ImageCaptionDataset loading complete with {len(self.data)} samples")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path, caption = self.data[idx]
        
        # If we're in placeholder mode, just return a placeholder image
        if self.use_placeholder_images:
            return self._get_placeholder_image(), caption
        
        # Load image
        img_path = os.path.join(self.image_dir, img_path)
        
        # Only print debug info for the first few images
        debug_print = idx < 5
        
        try:
            if debug_print:
                print(f"DEBUG: Attempting to load image {idx} from {img_path}")
            
            # Check if file exists
            if not os.path.exists(img_path):
                if debug_print:
                    print(f"WARNING: Image not found at {img_path}")
                
                # Try without extension
                base_name = os.path.splitext(os.path.basename(img_path))[0]
                candidates = []
                
                # Look for any file starting with the base name in the image directory
                if os.path.exists(self.image_dir):
                    for file in os.listdir(self.image_dir):
                        if file.startswith(base_name) or file.startswith(f"{(int(base_name.split('_')[-1])):d}"):
                            candidates.append(os.path.join(self.image_dir, file))
                            
                if candidates:
                    if debug_print:
                        print(f"Found alternative image candidates: {candidates[0]}")
                    img_path = candidates[0]
                else:
                    if debug_print:
                        print(f"No alternative image found, using placeholder")
                    return self._get_placeholder_image(), caption
            
            # Try to open the image
            image = Image.open(img_path).convert('RGB')
            if debug_print:
                print(f"Successfully loaded image with size {image.size}")
        except Exception as e:
            if debug_print:
                print(f"ERROR loading image {img_path}: {e}")
            # Return a placeholder image if the image cannot be loaded
            return self._get_placeholder_image(), caption
        
        if self.transform:
            image = self.transform(image)
        
        return image, caption
    
    def _get_placeholder_image(self):
        """Return a placeholder image when the actual image cannot be loaded."""
        placeholder = Image.new('RGB', (224, 224), color='black')
        if self.transform:
            placeholder = self.transform(placeholder)
        return placeholder
    
    @staticmethod
    def collate_fn(batch):
        """Custom collate function for handling image-caption pairs."""
        images = [item[0] for item in batch]
        captions = [item[1] for item in batch]
        return images, captions


class ParallelImageTextDataset(Dataset):
    """
    Dataset for parallel image-text pairs, where each image has corresponding text in two languages.
    Used for training the image-to-text generation in multiple languages.
    """
    def __init__(
        self,
        image_dir: str,
        caption_file_src: str,
        caption_file_tgt: str = None,
        transform=None,
        max_dataset_samples: int = None,
        src_lang: str = "en",
        tgt_lang: str = "de"
    ):
        """
        Args:
            image_dir (str): Directory with images
            caption_file_src (str): File with source language captions
            caption_file_tgt (str, optional): File with target language captions. If None, 
                                             assumes caption_file_src has both languages
            transform (callable, optional): Transform to apply to images
            max_dataset_samples (int, optional): Maximum number of samples to use
            src_lang (str): Source language code
            tgt_lang (str): Target language code
        """
        super(ParallelImageTextDataset, self).__init__()
        
        self.image_dir = image_dir
        self.transform = transform
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        
        print(f"\nDEBUG: Loading ParallelImageTextDataset")
        print(f"DEBUG: Source caption file: {caption_file_src}")
        print(f"DEBUG: Target caption file: {caption_file_tgt if caption_file_tgt else 'None (using source file)'}")
        print(f"DEBUG: Image directory: {image_dir}")
        print(f"DEBUG: Source language: {src_lang}, Target language: {tgt_lang}")
        
        # Load source captions
        self.data = []
        
        if caption_file_tgt is None:
            print(f"DEBUG: Using single file for both source and target captions")
            # Assume single file with both source and target captions
            if caption_file_src.endswith('.tsv'):
                print(f"DEBUG: Loading TSV caption file with both languages")
                # Format: image_path \t source_caption \t target_caption
                with open(caption_file_src, 'r', encoding='utf-8') as f:
                    for i, line in enumerate(f):
                        parts = line.strip().split('\t')
                        if len(parts) >= 3:
                            img_path, src_caption, tgt_caption = parts[0], parts[1], parts[2]
                            self.data.append((img_path, src_caption, tgt_caption))
                            if i < 3:
                                print(f"DEBUG: Sample {i}:")
                                print(f"  Image: {img_path}")
                                print(f"  Source ({src_lang}): {src_caption[:50]}...")
                                print(f"  Target ({tgt_lang}): {tgt_caption[:50]}...")
                print(f"DEBUG: Loaded {len(self.data)} triplets from TSV")
            elif caption_file_src.endswith('.csv'):
                print(f"DEBUG: Loading CSV caption file with both languages")
                df = pd.read_csv(caption_file_src)
                print(f"DEBUG: CSV columns: {df.columns.tolist()}")
                if 'image' in df.columns and src_lang in df.columns and tgt_lang in df.columns:
                    for i, (_, row) in enumerate(df.iterrows()):
                        self.data.append((row['image'], row[src_lang], row[tgt_lang]))
                        if i < 3:
                            print(f"DEBUG: Sample {i}:")
                            print(f"  Image: {row['image']}")
                            print(f"  Source ({src_lang}): {row[src_lang][:50]}...")
                            print(f"  Target ({tgt_lang}): {row[tgt_lang][:50]}...")
                    print(f"DEBUG: Loaded {len(self.data)} triplets from CSV")
                else:
                    raise ValueError(f"CSV file must have 'image', '{src_lang}', and '{tgt_lang}' columns")
        else:
            print(f"DEBUG: Using separate files for source and target captions")
            # Separate files for source and target captions
            src_captions = []
            tgt_captions = []
            img_paths = []
            
            # Load source captions
            if caption_file_src.endswith('.tsv'):
                print(f"DEBUG: Loading source captions from TSV")
                with open(caption_file_src, 'r', encoding='utf-8') as f:
                    for i, line in enumerate(f):
                        if '\t' in line:
                            img_path, caption = line.strip().split('\t', 1)
                            img_paths.append(img_path)
                            src_captions.append(caption)
                            if i < 3:
                                print(f"DEBUG: Source sample {i}: Image={img_path}, Caption={caption[:50]}...")
                print(f"DEBUG: Loaded {len(src_captions)} source captions")
            elif caption_file_src.endswith('.txt'):
                print(f"DEBUG: Loading source captions from TXT")
                with open(caption_file_src, 'r', encoding='utf-8') as f:
                    for i, line in enumerate(f):
                        parts = line.strip().split(' ', 1)
                        if len(parts) == 2:
                            img_path, caption = parts
                            img_paths.append(img_path)
                            src_captions.append(caption)
                            if i < 3:
                                print(f"DEBUG: Source sample {i}: Image={img_path}, Caption={caption[:50]}...")
                print(f"DEBUG: Loaded {len(src_captions)} source captions")
            elif os.path.basename(caption_file_src).startswith('train.') or os.path.basename(caption_file_src).startswith('test_') or os.path.basename(caption_file_src).startswith('val_'):
                print(f"DEBUG: Loading Multi30k plain text caption file for source")
                # Multi30k format - one caption per line, no image reference
                with open(caption_file_src, 'r', encoding='utf-8') as f:
                    for i, line in enumerate(f):
                        img_name = f"COCO_train_{(i+1):012d}.jpg"
                        caption = line.strip()
                        img_paths.append(img_name)
                        src_captions.append(caption)
                        if i < 3:
                            print(f"DEBUG: Source sample {i}: Image={img_name}, Caption={caption[:50]}...")
                print(f"DEBUG: Loaded {len(src_captions)} source captions from Multi30k plain text")
            else:
                print(f"DEBUG: Loading source captions from CSV")
                # CSV format
                try:
                    df = pd.read_csv(caption_file_src)
                    if 'image' in df.columns and self.src_lang in df.columns:
                        for i, (_, row) in enumerate(df.iterrows()):
                            img_paths.append(row['image'])
                            src_captions.append(row[self.src_lang])
                            if i < 3:
                                print(f"DEBUG: Source sample {i}: Image={row['image']}, Caption={row[self.src_lang][:50]}...")
                    else:
                        raise ValueError(f"CSV file must have 'image' and '{self.src_lang}' columns")
                    print(f"DEBUG: Loaded {len(src_captions)} source captions from CSV")
                except Exception as e:
                    print(f"DEBUG: Error loading CSV source: {e}")
                    print(f"DEBUG: Falling back to plain text format for source")
                    with open(caption_file_src, 'r', encoding='utf-8') as f:
                        for i, line in enumerate(f):
                            img_name = f"COCO_train_{(i+1):012d}.jpg"
                            caption = line.strip()
                            img_paths.append(img_name)
                            src_captions.append(caption)
                            if i < 3:
                                print(f"DEBUG: Source sample {i}: Image={img_name}, Caption={caption[:50]}...")
                    print(f"DEBUG: Loaded {len(src_captions)} source captions from plain text")
            
            # Load target captions
            if caption_file_tgt.endswith('.tsv'):
                print(f"DEBUG: Loading target captions from TSV")
                with open(caption_file_tgt, 'r', encoding='utf-8') as f:
                    for i, line in enumerate(f):
                        if '\t' in line:
                            _, caption = line.strip().split('\t', 1)
                            tgt_captions.append(caption)
                            if i < 3:
                                print(f"DEBUG: Target sample {i}: Caption={caption[:50]}...")
                print(f"DEBUG: Loaded {len(tgt_captions)} target captions")
            elif caption_file_tgt.endswith('.txt'):
                print(f"DEBUG: Loading target captions from TXT")
                with open(caption_file_tgt, 'r', encoding='utf-8') as f:
                    for i, line in enumerate(f):
                        parts = line.strip().split(' ', 1)
                        if len(parts) == 2:
                            _, caption = parts
                            tgt_captions.append(caption)
                            if i < 3:
                                print(f"DEBUG: Target sample {i}: Caption={caption[:50]}...")
                print(f"DEBUG: Loaded {len(tgt_captions)} target captions")
            elif os.path.basename(caption_file_tgt).startswith('train.') or os.path.basename(caption_file_tgt).startswith('test_') or os.path.basename(caption_file_tgt).startswith('val_'):
                print(f"DEBUG: Loading Multi30k plain text caption file for target")
                # Multi30k format - one caption per line, no image reference
                with open(caption_file_tgt, 'r', encoding='utf-8') as f:
                    for i, line in enumerate(f):
                        caption = line.strip()
                        tgt_captions.append(caption)
                        if i < 3:
                            print(f"DEBUG: Target sample {i}: Caption={caption[:50]}...")
                print(f"DEBUG: Loaded {len(tgt_captions)} target captions from Multi30k plain text")
            else:
                print(f"DEBUG: Loading target captions from CSV")
                try:
                    df = pd.read_csv(caption_file_tgt)
                    if self.tgt_lang in df.columns:
                        for i, (_, row) in enumerate(df.iterrows()):
                            tgt_captions.append(row[self.tgt_lang])
                            if i < 3:
                                print(f"DEBUG: Target sample {i}: Caption={row[self.tgt_lang][:50]}...")
                    else:
                        raise ValueError(f"CSV file must have '{self.tgt_lang}' column")
                    print(f"DEBUG: Loaded {len(tgt_captions)} target captions from CSV")
                except Exception as e:
                    print(f"DEBUG: Error loading CSV target: {e}")
                    print(f"DEBUG: Falling back to plain text format for target")
                    with open(caption_file_tgt, 'r', encoding='utf-8') as f:
                        for i, line in enumerate(f):
                            caption = line.strip()
                            tgt_captions.append(caption)
                            if i < 3:
                                print(f"DEBUG: Target sample {i}: Caption={caption[:50]}...")
                    print(f"DEBUG: Loaded {len(tgt_captions)} target captions from plain text")
            
            # Check that we have the same number of src and tgt captions
            if len(src_captions) != len(tgt_captions):
                raise ValueError(f"Number of source captions ({len(src_captions)}) doesn't match number of target captions ({len(tgt_captions)})")
                
            # Create data
            for i in range(len(src_captions)):
                self.data.append((img_paths[i], src_captions[i], tgt_captions[i]))
        
        # Apply max samples limit
        if max_dataset_samples is not None and max_dataset_samples < len(self.data):
            random.seed(42)  # For reproducibility
            ix = random.sample(range(0, len(self.data)), max_dataset_samples)
            self.data = [self.data[i] for i in ix]
            print(f"DEBUG: Limited dataset to {len(self.data)} samples")
        
        # Verify image paths
        for i, (img_path, _, _) in enumerate(self.data[:5]):
            full_path = os.path.join(self.image_dir, img_path)
            exists = os.path.exists(full_path)
            print(f"DEBUG: Image {i} path check: {full_path} - Exists: {exists}")
            
        print(f"DEBUG: ParallelImageTextDataset loading complete with {len(self.data)} samples")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path, src_caption, tgt_caption = self.data[idx]
        
        # Load image
        img_path = os.path.join(self.image_dir, img_path)
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            logging.error(f"Error loading image {img_path}: {e}")
            # Return a placeholder image if the image cannot be loaded
            image = Image.new('RGB', (224, 224), color='black')
        
        if self.transform:
            image = self.transform(image)
        
        return image, src_caption, tgt_caption
    
    @staticmethod
    def collate_fn(batch):
        """Custom collate function for handling image-text pairs."""
        images = [item[0] for item in batch]
        src_captions = [item[1] for item in batch]
        tgt_captions = [item[2] for item in batch]
        return images, src_captions, tgt_captions