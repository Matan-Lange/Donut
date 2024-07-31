import os
import json
from typing import Any, List, Tuple
import torch
from torch.utils.data import Dataset
from PIL import Image
import re
import json
import dicttoxml
from xml.dom.minidom import parseString


class DonutDatasetFineTune(Dataset):
    """add docs"""

    def __init__(
            self,
            dataset_path: str,
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

        self.dataset = self.load_dataset(os.path.join(dataset_path, split))
        self.dataset_length = len(self.dataset)

    def json_to_xml(self, dict_obj):
        # Convert dictionary to XML
        xml_bytes = dicttoxml.dicttoxml(dict_obj, custom_root='root', attr_type=False)

        # Parse the XML bytes to a string
        xml_str = parseString(xml_bytes).toprettyxml(indent="  ")

        # Remove the XML declaration
        xml_pretty_str = '\n'.join(xml_str.split('\n')[1:])

        # Remove newline characters
        xml_str = xml_pretty_str.replace('\n', '')

        # Remove spaces between tags
        xml_str = re.sub(r'>\s+<', '><', xml_str)

        return xml_str, xml_pretty_str

    def load_dataset(self, dataset_path):
        dataset = []
        # may need to add encoding='utf-8' for hebrew
        with open(os.path.join(dataset_path, 'metadata.jsonl'), 'r', encoding='utf-8') as file:
            for line in file:
                data_point = json.loads(line)
                img_path = os.path.join(dataset_path, data_point['image_name'].split('\\')[-1])

                data_point.pop('image_name')

                text_sequence, pretty_sequence = self.json_to_xml(data_point)
                dataset.append(
                    {'img_path': img_path, 'text_sequence': text_sequence, "pretty_sequence": pretty_sequence}
                )
        return dataset

    def __len__(self) -> int:
        return self.dataset_length

    def __getitem__(self, idx: int):
        """
        """
        sample = self.dataset[idx]
        img = Image.open(sample["img_path"])

        # inputs
        pixel_values = self.processor(img, random_padding=self.split == "train", return_tensors="pt").pixel_values
        pixel_values = pixel_values.squeeze()

        # targets
        target_sequence = sample['text_sequence']
        pretty_sequence = sample['pretty_sequence']
        input_ids = self.processor.tokenizer(
            self.processor.tokenizer.bos_token + " " + target_sequence + " " + self.processor.tokenizer.eos_token,
            add_special_tokens=False,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )["input_ids"].squeeze(0)

        labels = input_ids.clone()
        labels[
            labels == self.processor.tokenizer.pad_token_id] = self.ignore_id  # model doesn't need to predict pad token
        return pixel_values, labels, pretty_sequence  # returns pixels labels and the text

