import torch
from flask import Flask, request, jsonify
from model_new import build_transformer
from trainandgetdata import get_config, greedy_decode, load_dataset, get_or_build_tokenizer, latest_weights_file_path

app = Flask(__name__)

# Load model configuration
config = get_config()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Prepare tokenizers and dataset (you can load an empty dataset here)
ds_raw = load_dataset(f"{config['datasource']}", f"{config['lang_src']}-{config['lang_tgt']}", split='train')
tokenizer_src = get_or_build_tokenizer(config, ds_raw, config['lang_src'])
tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config['lang_tgt'])

# Load or initialize model
model = build_transformer(
    src_vocab_size=tokenizer_src.get_vocab_size(),
    tgt_vocab_size=tokenizer_tgt.get_vocab_size(),
    src_seq_len=config['seq_len'],
    tgt_seq_len=config['seq_len'],
    d_model=config['d_model']
).to(device)


# Load pre-trained model weights if available
model_filename = latest_weights_file_path(config)
if model_filename:
    print(f"Loading pre-trained model: {model_filename}")
    state = torch.load(model_filename, map_location=device)
    model.load_state_dict(state['model_state_dict'])
else:
    print("No pre-trained model found, initializing new model.")

model.eval()  # Set model to evaluation mode


# Function to process input sentence and make a prediction
def translate_sentence(sentence):
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
    
    return predicted_sentence


# Define a route for the translation API
@app.route('/predict', methods=['POST'])
def predict():
    # Get input data (assuming JSON input with a "sentence" key)
    data = request.json
    sentence = data.get("sentence")
    
    if not sentence:
        return jsonify({"error": "No sentence provided"}), 400

    # Translate the input sentence using the model
    predicted_sentence = translate_sentence(sentence)
    
    return jsonify({"input": sentence, "predicted_translation": predicted_sentence})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
