from transformers import VisionEncoderDecoderConfig
from transformers import DonutProcessor, VisionEncoderDecoderModel


def load_he_model(repo_name, hf_token):
    parsing_tokens = ['<invoice>', '</invoice>', '<client>', '</client>', '<address>', '</address>', '<name>',
                      '</name>',
                      '<date>', '</date>', '<items> </items>', '<item>' '</item>', '<price>', '</price>', '<quantity>',
                      '</quantity>', '<number>', '</number>', '<total>', '</total>', '<root>', '</root>']

    config = VisionEncoderDecoderConfig.from_pretrained(repo_name, use_auth_token=hf_token)

    processor = DonutProcessor.from_pretrained(repo_name, use_auth_token=hf_token)
    model = VisionEncoderDecoderModel.from_pretrained(repo_name, config=config, use_auth_token=hf_token)
    processor.tokenizer.add_tokens(parsing_tokens)
    model.decoder.resize_token_embeddings(len(processor.tokenizer))

    return processor, model


from huggingface_hub import HfApi, HfFolder

# Get the Hugging Face API token
api = HfApi()
token = HfFolder.get_token()
repo_name = "jjjlangem/He-Donut"
processor, model = load_he_model(repo_name, token)

print(model)
print(processor.tokenizer)
