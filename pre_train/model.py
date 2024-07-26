from transformers import VisionEncoderDecoderConfig
from transformers import DonutProcessor, VisionEncoderDecoderModel, AutoTokenizer


# Todo - get new toknizer as arg
def load_base_model():
    """
    Loads and configures the Donut model for use with Hebrew text.

    This function performs the following steps:
    1. Sets the image size and maximum length parameters for the model.
    2. Loads the pre-trained Donut model and updates the image size configuration for the encoder.
    3. Updates the maximum length configuration for the decoder.
    4. Loads the pre-trained Donut processor.
    5. Loads a pre-trained Hebrew tokenizer.
    6. Adds a beginning-of-sequence (bos) token to the tokenizer.
    7. Replaces the processor's tokenizer with the new Hebrew tokenizer.
    8. Adjusts the model's embeddings to accommodate the new tokenizer.
    9. Updates the model configuration to use the new tokenizer's padding and bos tokens.

    Returns:
        processor: The configured DonutProcessor with the new Hebrew tokenizer.
        model: The configured VisionEncoderDecoderModel with updated settings for the Hebrew tokenizer.
    """
    image_size = [1280, 960]
    max_length = 768

    # update image_size of the encoder
    # during pre-training, a larger image size was used
    config = VisionEncoderDecoderConfig.from_pretrained("naver-clova-ix/donut-base")
    config.encoder.image_size = image_size  # (height, width)
    # update max_length of the decoder (for generation)
    config.decoder.max_length = max_length

    processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base")
    model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base", config=config)

    # use hebrew tokenzier
    tokenizer = AutoTokenizer.from_pretrained('Helsinki-NLP/opus-mt-en-he')

    # add bos_token
    tokenizer.add_special_tokens({'bos_token': '<s>'})
    # replace tokenizer in processor
    processor.tokenizer = tokenizer

    # adjusted embbedings to new tokenizer len
    model.decoder.resize_token_embeddings(len(processor.tokenizer))

    # change model config to work with new tokenizer
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.decoder_start_token_id = processor.tokenizer.bos_token_id

    return processor, model



