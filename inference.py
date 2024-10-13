import torch
from model_new import build_transformer
from trainandgetdata import get_config, greedy_decode, load_dataset, get_or_build_tokenizer, latest_weights_file_path


# Function to load a trained model or create a new one if no weights are found
def load_or_initialize_model(config, tokenizer_src, tokenizer_tgt):
    vocab_src_len = tokenizer_src.get_vocab_size()
    vocab_tgt_len = tokenizer_tgt.get_vocab_size()

    model = build_transformer(
        vocab_src_len,
        vocab_tgt_len,
        config['seq_len'],
        config['seq_len'],
        d_model=config['d_model']
    )

    model_filename = latest_weights_file_path(config)
    if model_filename:
        print(f"Loading pre-trained model: {model_filename}")
        state = torch.load(model_filename, map_location='cpu')
        model.load_state_dict(state['model_state_dict'])
    else:
        print("No pre-trained model found, initializing new model.")

    return model

# Function to test the model with a sample input sentence
def test_model_with_sentence(model, sentence, tokenizer_src, tokenizer_tgt, config, device):
    model.eval()

    # Tokenize and encode the input sentence
    input_tokens = tokenizer_src.encode(sentence).ids
    input_tensor = torch.tensor([input_tokens], dtype=torch.long, device=device)

    # Create the source mask
    src_mask = (input_tensor != tokenizer_src.token_to_id("[PAD]")).unsqueeze(0).unsqueeze(0).to(device)

    # Generate the output using greedy decoding
    predicted_tokens = greedy_decode(
        model,
        input_tensor,
        src_mask,
        tokenizer_src,
        tokenizer_tgt,
        config['seq_len'],
        device
    )

    # Decode the predicted tokens into a sentence
    predicted_sentence = tokenizer_tgt.decode(predicted_tokens.detach().cpu().numpy())

    print(f"Input: {sentence}")
    print(f"Predicted Translation: {predicted_sentence}")

# Testing the model with a sample sentence
if __name__ == '__main__':
    config = get_config()  # Load config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Prepare tokenizers and dataset (you can load an empty dataset here)
    ds_raw = load_dataset(f"{config['datasource']}", f"{config['lang_src']}-{config['lang_tgt']}", split='train')
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config['lang_tgt'])

    # Load or initialize model
    model = load_or_initialize_model(config, tokenizer_src, tokenizer_tgt).to(device)

    # Test with a sample input sentence
    test_sentence = "This is a test sentence in English."  
    test_model_with_sentence(model, test_sentence, tokenizer_src, tokenizer_tgt, config, device)
