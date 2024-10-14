from flask import Flask, request, jsonify, render_template
import torch
from model_new import build_transformer
from trainandgetdata import get_config, greedy_decode, load_dataset, get_or_build_tokenizer, latest_weights_file_path

app = Flask(__name__)

config = get_config()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ds_raw = load_dataset(f"{config['datasource']}", f"{config['lang_src']}-{config['lang_tgt']}", split='train')
tokenizer_src = get_or_build_tokenizer(config, ds_raw, config['lang_src'])
tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config['lang_tgt'])

model = build_transformer(
    src_vocab_size=tokenizer_src.get_vocab_size(),
    tgt_vocab_size=tokenizer_tgt.get_vocab_size(),
    src_seq_len=config['seq_len'],
    tgt_seq_len=config['seq_len'],
    d_model=config['d_model']
).to(device)

model_filename = latest_weights_file_path(config)
if model_filename:
    print(f"Loading pre-trained model: {model_filename}")
    state = torch.load(model_filename, map_location=device)
    model.load_state_dict(state['model_state_dict'])
else:
    print("No pre-trained model found, initializing new model.")

model.eval()  


def translate_sentence(sentence):
    input_tokens = tokenizer_src.encode(sentence).ids
    input_tensor = torch.tensor([input_tokens], dtype=torch.long, device=device)

    src_mask = (input_tensor != tokenizer_src.token_to_id("[PAD]")).unsqueeze(0).unsqueeze(0).to(device)

    predicted_tokens = greedy_decode(
        model,
        input_tensor,
        src_mask,
        tokenizer_src,
        tokenizer_tgt,
        config['seq_len'],
        device
    )

    predicted_sentence = tokenizer_tgt.decode(predicted_tokens.detach().cpu().numpy())
    
    return predicted_sentence


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    sentence = data.get("sentence")
    
    if not sentence:
        return jsonify({"error": "No sentence provided"}), 400

    predicted_sentence = translate_sentence(sentence)
    
    return jsonify({"input": sentence, "predicted_translation": predicted_sentence})


@app.route('/')
def index():
    return render_template('index.html')  


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
