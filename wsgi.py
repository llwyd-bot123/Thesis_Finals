import model
from flask import Flask, render_template, request, redirect, url_for, jsonify
import sys
import os
sys.path.insert(0, os.getcwd())

app = Flask(__name__)


# @app.route('/', methods=['GET', 'POST'])
# def index():
#     return render_template('home.html')


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == "POST":
        sample_input = request.form.get('userDomain')
        dir_hash = sample_input
        model.model_dict, model.source_tokenizer, model.target_tokenizer, model._, model.encoder_model, model.decoder_model = model.load_saved_model(
            dir_hash)
        input_text = request.form['textVal']
        if dir_hash == "nstd-std":
            translated_output = model.lr(input_text)
        elif dir_hash == "std-nstd":
            translated_output = model.rl(input_text)
        else:
            translated_output = "None"
        return jsonify(translated_output)
    return render_template('home.html')


if __name__ == "__main__":
    app.run(debug=True)
