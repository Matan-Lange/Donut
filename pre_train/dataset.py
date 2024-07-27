import os
import json
from typing import Any, List, Tuple
import torch
from torch.utils.data import Dataset
from PIL import Image


class DonutDataset(Dataset):
    """
    PyTorch Dataset for Donut.

    Each row, consists of image path(png/jpg/jpeg) and gt data (json/jsonl/txt),
    and it will be converted into pixel_values (vectorized image) and labels (input_ids of the tokenized string).

    Args:
        dataset_name_or_path: path containing image files and metadata.jsonl
        max_length: the max number of tokens for the target sequences
        split: whether to load "train", "validation" or "test" split
        ignore_id: ignore_index for torch.nn.CrossEntropyLoss
    """

    def __init__(
            self,
            dataset_name_or_path: str,
            max_length: int,
            processor,
            split: str = "train",
            ignore_id: int = -100
    ):
        super().__init__()

        self.max_length = max_length
        self.split = split
        self.ignore_id = ignore_id
        self.processor = processor

        self.dataset = self.load_dataset(dataset_name_or_path)
        self.dataset_length = len(self.dataset)

    def load_dataset(self, dataset_name_or_path):
        dataset = []
        #may need to add encoding='utf-8' for hebrew
        with open(os.path.join(dataset_name_or_path, 'metadata.jsonl'), 'r',encoding='utf-8') as file:
            for line in file:
                data_point = json.loads(line)
                img_path = os.path.join(dataset_name_or_path, data_point['file_name'])
                text_sequence = json.loads(data_point['ground_truth'])['gt_parse']['text_sequence']
                dataset.append({'img_path': img_path, 'text_sequence': text_sequence})
        return dataset

    def __len__(self) -> int:
        return self.dataset_length

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Load image from image_path of given dataset_path and convert into input_tensor and labels
        Convert gt data into input_ids (tokenized string)
        Returns:
            input_tensor : preprocessed image
            input_ids : tokenized gt_data
            labels : masked labels (model doesn't need to predict prompt and pad token)
        """
        sample = self.dataset[idx]
        img = Image.open(sample["img_path"])

        # inputs
        pixel_values = self.processor(img, random_padding=self.split == "train", return_tensors="pt").pixel_values
        pixel_values = pixel_values.squeeze()

        # targets
        target_sequence = sample['text_sequence']
        input_ids = self.processor.tokenizer(
            self.processor.tokenizer.bos_token + " " + target_sequence + " " + self.processor.tokenizer.eos_token,
            add_special_tokens=False,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            )["input_ids"].squeeze(0)

        labels = input_ids.clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = self.ignore_id  # model doesn't need to predict pad token
        return pixel_values, labels, target_sequence  # returns pixels labels and the text
