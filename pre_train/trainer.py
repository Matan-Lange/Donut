import torch
from torch.utils.data import DataLoader
import mlflow
from nltk import edit_distance
import numpy as np
import re


class DonutModel(torch.nn.Module):
    def __init__(self, model):
        super(DonutModel, self).__init__()
        self.model = model

    def forward(self, pixel_values, labels=None):
        return self.model(pixel_values, labels=labels)


class Trainer:
    def __init__(self, config, processor, model, train_dataloader, val_dataloader=None):
        self.config = config
        self.processor = processor
        self.model = DonutModel(model)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.get("lr", 1e-4))

    def train_step(self, batch):
        self.model.train()
        pixel_values, labels, _ = batch
        pixel_values, labels = pixel_values.to(self.device), labels.to(self.device)

        self.optimizer.zero_grad()
        outputs = self.model(pixel_values, labels=labels)
        loss = outputs.loss
        loss.backward()
        self.optimizer.step()

        mlflow.log_metric("train_loss", loss.item())
        print(f"train loss: {loss.item()}")

        return loss.item()

    def validation_step(self, batch):
        self.model.eval()
        pixel_values, labels, answers = batch
        pixel_values, labels = pixel_values.to(self.device), labels.to(self.device)

        batch_size = pixel_values.shape[0]
        decoder_input_ids = torch.full((batch_size, 1), self.model.model.config.decoder_start_token_id,
                                       device=self.device)

        outputs = self.model.model.generate(pixel_values,
                                            decoder_input_ids=decoder_input_ids,
                                            max_length=self.config.get("max_length", 768),
                                            early_stopping=True,
                                            pad_token_id=self.processor.tokenizer.pad_token_id,
                                            eos_token_id=self.processor.tokenizer.eos_token_id,
                                            use_cache=True,
                                            num_beams=1,
                                            bad_words_ids=[[self.processor.tokenizer.unk_token_id]],
                                            return_dict_in_generate=True)

        predictions = []
        for seq in self.processor.tokenizer.batch_decode(outputs.sequences):
            seq = seq.replace(self.processor.tokenizer.eos_token, "").replace(self.processor.tokenizer.pad_token, "")
            seq = re.sub(r"<.*?>", "", seq, count=1).strip()
            predictions.append(seq)

        scores = []
        for pred, answer in zip(predictions, answers):
            answer = answer.replace(self.processor.tokenizer.eos_token, "")
            scores.append(edit_distance(pred, answer) / max(len(pred), len(answer)))

            if self.config.get("verbose", False) and len(scores) == 1:
                print(f"Prediction: {predictions}")
                print(f"    Answer: {answer}")
                print(f" Normed ED: {scores[0]}")

        avg_score = np.mean(scores)
        mlflow.log_metric("val_edit_distance", avg_score)

        return avg_score

    def train(self, epochs, validate_after_each_step=False):
        for epoch in range(epochs):
            train_losses = []
            for batch in self.train_dataloader:
                train_loss = self.train_step(batch)
                train_losses.append(train_loss)

                if validate_after_each_step and self.val_dataloader:
                    val_batch = next(iter(self.val_dataloader))
                    self.validation_step(val_batch)

            if not validate_after_each_step and self.val_dataloader:
                val_scores = []
                for batch in self.val_dataloader:
                    val_score = self.validation_step(batch)
                    val_scores.append(val_score)
                avg_val_score = np.mean(val_scores)
                print(f"Epoch {epoch + 1}, Validation Edit Distance: {avg_val_score}")

            avg_train_loss = np.mean(train_losses)
            print(f"Epoch {epoch + 1}, Train Loss: {avg_train_loss}")



