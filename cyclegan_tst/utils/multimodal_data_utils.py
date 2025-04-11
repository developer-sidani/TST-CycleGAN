import os
import torch
import pickle as pkl
from PIL import Image
from typing import List, Dict, Tuple, Union, Optional
import logging
from tqdm import tqdm


class Multi30kDataLoader:
    """
    Data loader for Multi30k dataset that handles multimodal data
    """

    def __init__(
            self,
            data_dir: str,
            langs: List[str] = ["en", "de"],
            test_sets: List[Tuple[str, str]] = [("2016", "flickr")],
            image_dir: str = "data/multi30k/images",
            cache_dir: str = "data/multi30k/cache",
            max_samples: int = None,
            image_processor=None,
            text_processor=None
    ):
        """
        Initialize the data loader

        Args:
            data_dir: Base directory for Multi30k data
            langs: List of languages to load
            test_sets: List of test sets to load as (year, name)
            image_dir: Directory containing images
            cache_dir: Directory to cache processed data
            max_samples: Maximum number of samples to load (None for all)
            image_processor: Function to process images
            text_processor: Function to process text
        """
        self.data_dir = data_dir
        self.langs = langs
        self.test_sets = test_sets
        self.image_dir = image_dir
        self.cache_dir = cache_dir
        self.max_samples = max_samples
        self.image_processor = image_processor
        self.text_processor = text_processor

        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)

        # Processed data will be stored here
        self.train_data = {}
        self.test_data = {}

        # Load data
        self._load_data()

    def _load_data(self):
        """Load text and image data from Multi30k"""
        # Load train data for all languages
        self.train_data["text"] = {}
        for lang in self.langs:
            train_file = os.path.join(self.data_dir, f"train.{lang}")
            cache_file = os.path.join(self.cache_dir, f"train.{lang}.pkl")

            if os.path.exists(cache_file):
                # Load cached data
                with open(cache_file, "rb") as f:
                    self.train_data["text"][lang] = pkl.load(f)
            else:
                # Load and process data
                with open(train_file, "r") as f:
                    text_data = f.read().splitlines()

                if self.max_samples:
                    text_data = text_data[:self.max_samples]

                # Process text if processor is provided
                if self.text_processor:
                    processed_data = []
                    for text in tqdm(text_data, desc=f"Processing train.{lang}"):
                        processed_data.append(self.text_processor(text))
                    self.train_data["text"][lang] = processed_data
                else:
                    self.train_data["text"][lang] = text_data

                # Cache processed data
                with open(cache_file, "wb") as f:
                    pkl.dump(self.train_data["text"][lang], f)

        # Load image splits
        train_image_file = os.path.join(self.data_dir, "text/data/task1/image_splits/train.txt")
        with open(train_image_file, "r") as f:
            self.train_data["image_paths"] = f.read().splitlines()

        if self.max_samples:
            self.train_data["image_paths"] = self.train_data["image_paths"][:self.max_samples]

        # Load and process train images if processor is provided
        if self.image_processor:
            cache_file = os.path.join(self.cache_dir, "train_images.pkl")
            if os.path.exists(cache_file):
                # Load cached image data
                self.train_data["images"] = torch.load(cache_file)
            else:
                # Process images
                self.train_data["images"] = []
                for img_path in tqdm(self.train_data["image_paths"], desc="Processing train images"):
                    full_path = os.path.join(self.image_dir, "train", img_path)
                    try:
                        img = Image.open(full_path).convert("RGB")
                        processed_img = self.image_processor(img)
                        self.train_data["images"].append(processed_img)
                    except Exception as e:
                        logging.warning(f"Error processing {full_path}: {e}")
                        self.train_data["images"].append(None)

                # Filter out None values
                valid_indices = [i for i, img in enumerate(self.train_data["images"]) if img is not None]
                self.train_data["images"] = [self.train_data["images"][i] for i in valid_indices]
                self.train_data["image_paths"] = [self.train_data["image_paths"][i] for i in valid_indices]
                for lang in self.langs:
                    self.train_data["text"][lang] = [self.train_data["text"][lang][i] for i in valid_indices]

                # Cache processed images
                torch.save(self.train_data["images"], cache_file)

        # Load test data for all test sets
        self.test_data = {}
        for year, name in self.test_sets:
            self.test_data[f"{year}_{name}"] = {"text": {}}

            # Load text data
            for lang in self.langs:
                test_file = os.path.join(self.data_dir, f"test_{year}_{name}.{lang}")
                cache_file = os.path.join(self.cache_dir, f"test_{year}_{name}.{lang}.pkl")

                if os.path.exists(cache_file):
                    # Load cached data
                    with open(cache_file, "rb") as f:
                        self.test_data[f"{year}_{name}"]["text"][lang] = pkl.load(f)
                else:
                    # Load and process data
                    with open(test_file, "r") as f:
                        text_data = f.read().splitlines()

                    # Process text if processor is provided
                    if self.text_processor:
                        processed_data = []
                        for text in tqdm(text_data, desc=f"Processing test_{year}_{name}.{lang}"):
                            processed_data.append(self.text_processor(text))
                        self.test_data[f"{year}_{name}"]["text"][lang] = processed_data
                    else:
                        self.test_data[f"{year}_{name}"]["text"][lang] = text_data

                    # Cache processed data
                    with open(cache_file, "wb") as f:
                        pkl.dump(self.test_data[f"{year}_{name}"]["text"][lang], f)

            # Load image splits
            test_image_file = os.path.join(
                self.data_dir, f"text/data/task1/image_splits/test_{year}_{name}.txt"
            )
            with open(test_image_file, "r") as f:
                self.test_data[f"{year}_{name}"]["image_paths"] = f.read().splitlines()

            # Load and process test images if processor is provided
            if self.image_processor:
                cache_file = os.path.join(self.cache_dir, f"test_{year}_{name}_images.pkl")
                if os.path.exists(cache_file):
                    # Load cached image data
                    self.test_data[f"{year}_{name}"]["images"] = torch.load(cache_file)
                else:
                    # Process images
                    self.test_data[f"{year}_{name}"]["images"] = []
                    img_dir = f"test_{year}_{name}"
                    for img_path in tqdm(
                            self.test_data[f"{year}_{name}"]["image_paths"],
                            desc=f"Processing test_{year}_{name} images"
                    ):
                        full_path = os.path.join(self.image_dir, img_dir, img_path)
                        try:
                            img = Image.open(full_path).convert("RGB")
                            processed_img = self.image_processor(img)
                            self.test_data[f"{year}_{name}"]["images"].append(processed_img)
                        except Exception as e:
                            logging.warning(f"Error processing {full_path}: {e}")
                            self.test_data[f"{year}_{name}"]["images"].append(None)

                    # Filter out None values
                    valid_indices = [i for i, img in enumerate(self.test_data[f"{year}_{name}"]["images"]) if
                                     img is not None]
                    self.test_data[f"{year}_{name}"]["images"] = [
                        self.test_data[f"{year}_{name}"]["images"][i] for i in valid_indices
                    ]
                    self.test_data[f"{year}_{name}"]["image_paths"] = [
                        self.test_data[f"{year}_{name}"]["image_paths"][i] for i in valid_indices
                    ]
                    for lang in self.langs:
                        self.test_data[f"{year}_{name}"]["text"][lang] = [
                            self.test_data[f"{year}_{name}"]["text"][lang][i] for i in valid_indices
                        ]

                    # Cache processed images
                    torch.save(self.test_data[f"{year}_{name}"]["images"], cache_file)

    def get_train_data(
            self,
            src_lang: str = "en",
            tgt_lang: str = "de",
            include_images: bool = True
    ) -> Tuple[List[str], List[str], Optional[List]]:
        """
        Get training data

        Args:
            src_lang: Source language
            tgt_lang: Target language
            include_images: Whether to include images

        Returns:
            Tuple of (source_texts, target_texts, images)
        """
        if src_lang not in self.langs or tgt_lang not in self.langs:
            raise ValueError(f"Languages must be in {self.langs}")

        src_texts = self.train_data["text"][src_lang]
        tgt_texts = self.train_data["text"][tgt_lang]

        if include_images and "images" in self.train_data:
            return src_texts, tgt_texts, self.train_data["images"]
        else:
            return src_texts, tgt_texts, None

    def get_test_data(
            self,
            test_set: Tuple[str, str],
            src_lang: str = "en",
            tgt_lang: str = "de",
            include_images: bool = True
    ) -> Tuple[List[str], List[str], Optional[List]]:
        """
        Get test data

        Args:
            test_set: Test set as (year, name)
            src_lang: Source language
            tgt_lang: Target language
            include_images: Whether to include images

        Returns:
            Tuple of (source_texts, target_texts, images)
        """
        year, name = test_set
        test_key = f"{year}_{name}"

        if test_key not in self.test_data:
            raise ValueError(f"Test set {test_key} not found")

        if src_lang not in self.langs or tgt_lang not in self.langs:
            raise ValueError(f"Languages must be in {self.langs}")

        src_texts = self.test_data[test_key]["text"][src_lang]
        tgt_texts = self.test_data[test_key]["text"][tgt_lang]

        if include_images and "images" in self.test_data[test_key]:
            return src_texts, tgt_texts, self.test_data[test_key]["images"]
        else:
            return src_texts, tgt_texts, None


class MultimodalDataset(torch.utils.data.Dataset):
    """
    Dataset for multimodal translation that includes text and images
    """

    def __init__(
            self,
            src_texts: List[str],
            tgt_texts: List[str],
            images: List = None,
            transform=None
    ):
        """
        Initialize dataset

        Args:
            src_texts: Source language texts
            tgt_texts: Target language texts
            images: Optional images
            transform: Optional transform to apply to images
        """
        self.src_texts = src_texts
        self.tgt_texts = tgt_texts
        self.images = images
        self.transform = transform

        # Validate data
        assert len(src_texts) == len(tgt_texts), "Source and target texts must have same length"
        if images is not None:
            assert len(src_texts) == len(images), "Texts and images must have same length"

    def __len__(self):
        return len(self.src_texts)

    def __getitem__(self, idx):
        sample = {
            "src_text": self.src_texts[idx],
            "tgt_text": self.tgt_texts[idx]
        }

        if self.images is not None:
            image = self.images[idx]
            if self.transform:
                image = self.transform(image)
            sample["image"] = image

        return sample


def create_dataloaders(
        data_loader: Multi30kDataLoader,
        src_lang: str = "en",
        tgt_lang: str = "de",
        test_set: Tuple[str, str] = ("2016", "flickr"),
        batch_size: int = 32,
        include_images: bool = True,
        shuffle_train: bool = True,
        num_workers: int = 4,
        pin_memory: bool = True
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Create DataLoaders for training and testing

    Args:
        data_loader: Multi30kDataLoader instance
        src_lang: Source language
        tgt_lang: Target language
        test_set: Test set as (year, name)
        batch_size: Batch size
        include_images: Whether to include images
        shuffle_train: Whether to shuffle training data
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory

    Returns:
        Tuple of (train_dataloader, test_dataloader)
    """
    # Get data
    train_src, train_tgt, train_images = data_loader.get_train_data(
        src_lang=src_lang,
        tgt_lang=tgt_lang,
        include_images=include_images
    )

    test_src, test_tgt, test_images = data_loader.get_test_data(
        test_set=test_set,
        src_lang=src_lang,
        tgt_lang=tgt_lang,
        include_images=include_images
    )

    # Create datasets
    train_dataset = MultimodalDataset(train_src, train_tgt, train_images)
    test_dataset = MultimodalDataset(test_src, test_tgt, test_images)

    # Create dataloaders
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=multimodal_collate_fn
    )

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=multimodal_collate_fn
    )

    return train_dataloader, test_dataloader


def multimodal_collate_fn(batch):
    """
    Collate function for multimodal batches

    Args:
        batch: List of samples

    Returns:
        Dictionary of batched data
    """
    src_texts = [item["src_text"] for item in batch]
    tgt_texts = [item["tgt_text"] for item in batch]

    result = {
        "src_texts": src_texts,
        "tgt_texts": tgt_texts
    }

    if "image" in batch[0]:
        images = [item["image"] for item in batch]
        if torch.is_tensor(images[0]):
            images = torch.stack(images)
        result["images"] = images

    return result