import torch
from torch.utils.data import DataLoader, DistributedSampler
import mlflow
from nltk import edit_distance
import numpy as np
import re
import os
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from dataset import DonutDatasetFineTune
from model import load_he_model


class TrainerDDP:
    def __init__(self, config, processor, model, train_dataloader, val_dataloader=None, rank=0, world_size=1):
        self.config = config
        self.processor = processor
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = iter(val_dataloader)
        self.device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.get("lr", 3e-5))
        self.scaler = torch.cuda.amp.GradScaler()
        self.global_step = 0
        self.rank = rank
        self.world_size = world_size
        self.gradient_clip_val = config.get("gradient_clip_val", 1.0)

        if self.world_size > 1:
            self.model = DDP(self.model, device_ids=[self.rank])

        mlflow.set_tracking_uri(config.get("mlflow_tracking_uri"))
        experiment_name = config.get("experiment_name")
        mlflow.set_experiment(experiment_name)

    def train_step(self, batch):
        self.model.train()
        pixel_values, labels, _ = batch
        pixel_values, labels = pixel_values.to(self.device), labels.to(self.device)

        self.optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            outputs = self.model(pixel_values, labels=labels)
            loss = outputs.loss

        self.scaler.scale(loss).backward()

        # Apply gradient clipping
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_val)

        self.scaler.step(self.optimizer)
        self.scaler.update()

        self.global_step += 1
        if self.rank == 0:
            mlflow.log_metric("train_loss", loss.item(), step=self.global_step)
            print(f"train loss: {loss.item()} at step {self.global_step}")

        return loss.item()

    def validation_step(self, batch):
        self.model.eval()
        pixel_values, labels, answers = batch
        pixel_values, labels = pixel_values.to(self.device), labels.to(self.device)

        batch_size = pixel_values.shape[0]
        # In PyTorch DDP setup, self.model is wrapped by DistributedDataParallel,
        # so the actual model is accessible through self.model.module.
        if self.world_size > 1:
            model = self.model.module
        else:
            model = self.model

        decoder_input_ids = torch.full((batch_size, 1), model.config.decoder_start_token_id,
                                       device=self.device)
        
        outputs = model.generate(pixel_values,
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
            seq = seq.replace(self.processor.tokenizer.eos_token, "").replace(self.processor.tokenizer.pad_token, "").replace(self.processor.tokenizer.bos_token, "")
            seq = re.sub(r"<.*?>", "", seq, count=1).strip()
            predictions.append(seq)

        scores = []
        for pred, answer in zip(predictions, answers):
            answer = answer.replace(self.processor.tokenizer.eos_token, "")
            score = edit_distance(pred, answer) / max(len(pred), len(answer))
            scores.append(score)

            if self.config.get("verbose", False) and self.rank == 0:
                print(f"Prediction: {pred}")
                print(f"    Answer: {answer}")
                print(f" Normed ED: {score}")

        avg_score = np.mean(scores)
        if self.rank == 0:
            mlflow.log_metric("val_edit_distance", avg_score, step=self.global_step)

        return avg_score

    def train(self, epochs):
        for epoch in range(epochs):
            train_losses = []
            for batch in self.train_dataloader:
                train_loss = self.train_step(batch)
                train_losses.append(train_loss)

                # Perform validation every 100 steps
                if self.global_step % 100 == 0 and self.val_dataloader:
                    val_batch = next(self.val_dataloader)
                    val_score = self.validation_step(val_batch)

            avg_train_loss = np.mean(train_losses)
            if self.rank == 0:
                print(f"Epoch {epoch + 1}, Train Loss: {avg_train_loss}")
                self.save_model(f'donut_model{epoch}.pt')

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup():
    dist.destroy_process_group()



def main(rank, 
         world_size,
         repo_name,
         data_set_path
        ):
    setup(rank, world_size)
    processor, model = load_he_model()


    train_dataset = DonutDatasetFineTune(os.path.join(data_set_path, 'train'), max_length=768, processor=processor)

    val_dataset = DonutDatasetFineTune(os.path.join(data_set_path, 'validation'), max_length=768, processor=processor)

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    train_dataloader = DataLoader(train_dataset, batch_size=4, sampler=train_sampler)

    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank)
    val_dataloader = DataLoader(val_dataset, batch_size=4, sampler=val_sampler)

    config = {
        "lr": 3e-5,
        "verbose": True,
        "gradient_clip_val": 1.0,
        'experiment_name': 'fine_tune_donut',
        "mlflow_tracking_uri": 'sqlite:///mlflow.db'  # add tracking uri here

    }

    trainer = TrainerDDP(config, processor, model, train_dataloader, val_dataloader, rank, world_size)
    trainer.train(3)

    cleanup()


if __name__ == "__main__":
    import torch.multiprocessing as mp

    world_size = 1
    if world_size > 1:
        mp.spawn(main, args=(world_size,), nprocs=world_size, join=True)
    else:
        main(0, world_size)
