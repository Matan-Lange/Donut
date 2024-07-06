from transformers import DonutProcessor, VisionEncoderDecoderConfig, VisionEncoderDecoderModel, M2M100Tokenizer
from transformers import MarianMTModel, MarianTokenizer
from transformers.modeling_outputs import BaseModelOutput
from torch.utils.data import DataLoader, Dataset, random_split
import torch
from PIL import Image
from faker import Faker
import torchvision.transforms as transforms
from PIL import ImageFile
from tqdm import tqdm
from transformers import M2M100ForConditionalGeneration, AdamW, PreTrainedModel
from transformers.modeling_outputs import Seq2SeqLMOutput
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.utils.rnn as rnn_utils
from torch.cuda.amp import GradScaler, autocast
import torch.nn.utils.prune as prune
import gc

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
scaler = GradScaler()


fake = Faker()
ImageFile.LOAD_TRUNCATED_IMAGES = True

class CustomDataset(Dataset):
    def __init__(self, dataset, processor):
        self.filepaths = [sample.filepath for sample in dataset]
        self.processor = processor

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        image_path = self.filepaths[idx]
        image = Image.open(image_path).convert("RGB")  
        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values.squeeze()
        text = fake.sentence(nb_words=10)  
        return pixel_values, text
    
# Training loop

def pretrain_donut(custom_model, train_dataloader, val_dataloader, tokenizer, lr=5e-5, num_epochs = 5, accumulation_steps = 10):

    optimizer = AdamW(custom_model.parameters(), lr=lr)  
    custom_model.to(device)
    for epoch in range(num_epochs):
        custom_model.train()
        total_loss = 0
        optimizer.zero_grad()

        for i, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch")):
            images, texts = batch

            # Move data to the device
            images = images.to(device, non_blocking=True)
            inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(device, non_blocking=True) for k, v in inputs.items()}

            # Forward pass
            with autocast():
                outputs = custom_model(pixel_values=images, decoder_input_ids=inputs['input_ids'], labels=inputs['input_ids'])

                # Compute loss
                loss = outputs.loss

            total_loss += loss.item()

            # Backpropagation
            scaler.scale(loss).backward()

            if (i + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            # Clear memory
            del images, texts, inputs, outputs, loss
            torch.cuda.empty_cache()
            gc.collect()

        if (i + 1) % accumulation_steps != 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {avg_loss:.4f}")

        # Validation loop
        custom_model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_dataloader:
                images, texts = batch

                # Move data to the device
                images = images.to(device, non_blocking=True)
                inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
                inputs = {k: v.to(device, non_blocking=True) for k, v in inputs.items()}

                # Forward pass
                with autocast():
                    outputs = custom_model(pixel_values=images, decoder_input_ids=inputs['input_ids'], labels=inputs['input_ids'])

                    # Compute loss
                    loss = outputs.loss

                val_loss += loss.item()

                # Clear memory
                del images, texts, inputs, outputs, loss
                torch.cuda.empty_cache()
                gc.collect()

        avg_val_loss = val_loss / len(val_dataloader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Validation Loss: {avg_val_loss:.4f}")

    print("Training completed.")