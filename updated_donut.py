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

swin_model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base")
marian_model_name = "Helsinki-NLP/opus-mt-en-he"
marian_tokenizer = MarianTokenizer.from_pretrained(marian_model_name)
marian_model = MarianMTModel.from_pretrained(marian_model_name)

# Reduce the number of decoder layers in MarianMT
class ReducedMarianMTModel(MarianMTModel):
    def __init__(self, config):
        super(ReducedMarianMTModel, self).__init__(config)
        self.model.decoder.layers = self.model.decoder.layers[:2]  # Keep only 2 decoder layers

reduced_marian_model = ReducedMarianMTModel(marian_model.config)

# Extract the encoder from the Swin Transformer
swin_encoder = swin_model.encoder

# Create an adapter to adjust the embedding size if necessary
class Adapter(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Adapter, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)

embedding_adapter = Adapter(input_dim=1024, output_dim=reduced_marian_model.config.d_model)

# Create the combined configuration
encoder_config = swin_model.encoder.config
decoder_config = reduced_marian_model.config
custom_config = VisionEncoderDecoderConfig.from_encoder_decoder_configs(encoder_config, decoder_config)

class CustomVisionEncoderDecoderModel(VisionEncoderDecoderModel):
    def __init__(self, config, encoder, decoder, adapter):
        super(CustomVisionEncoderDecoderModel, self).__init__(config, encoder, decoder)
        self.adapter = adapter

    def forward(self, pixel_values, decoder_input_ids, attention_mask=None, decoder_attention_mask=None, labels=None, **kwargs):
        encoder_outputs = self.encoder(pixel_values=pixel_values, return_dict=True)
        encoder_hidden_states = encoder_outputs.last_hidden_state
        adapted_hidden_states = self.adapter(encoder_hidden_states)
        
        # Create a BaseModelOutput to pass to the decoder
        encoder_outputs = BaseModelOutput(
            last_hidden_state=adapted_hidden_states,
            hidden_states=None,
            attentions=None
        )
        
        # Ensure decoder takes adapted hidden states properly
        decoder_outputs = self.decoder.model.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=adapted_hidden_states,
            encoder_attention_mask=attention_mask,
            return_dict=True,
            use_cache=False,
            **kwargs
        )
        
        lm_logits = self.decoder.lm_head(decoder_outputs.last_hidden_state) + self.decoder.final_logits_bias

        loss = None
        if labels is not None:
            shift_logits = lm_logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            pad_token_id = self.config.pad_token_id
            if pad_token_id is None:
                pad_token_id = self.config.decoder.pad_token_id
            loss_fct = nn.CrossEntropyLoss(ignore_index=pad_token_id)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_hidden_states,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

class CustomDonutProcessor(DonutProcessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.feature_extractor.size = {"height": 480, "width": 360}