import os
import json
import sqlite3
from transformers import GPT2Tokenizer, TFGPT2LMHeadModel
from flask import Flask, request, jsonify

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

app = Flask(__name__)

tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')
model = TFGPT2LMHeadModel.from_pretrained('distilgpt2')

tokenizer.pad_token = tokenizer.eos_token
_max_length = 50
_num_return_sequences = 1
_temperature = 0.7
_top_p = 0.9
_repetition_penalty = 1.2
_do_sample = True

@app.route("/generate", methods=["POST"])
def generate_text():
	data = request.json
	input_text = data.get("text", "")

	encoded_input = tokenizer(input_text, return_tensors='tf')
	outputs = model.generate(encoded_input['input_ids'],
				max_length = _max_length,
				num_return_sequences = _num_return_sequences,
				temperature = _temperature,
				top_p = _top_p,
				repetition_penalty = _repetition_penalty,
				do_sample = _do_sample,
				attention_mask=encoded_input['attention_mask'])

	generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

	conn = sqlite3.connect('history.db', check_same_thread=False)
	cursor = conn.cursor()
	cursor.execute('INSERT INTO messages (user_prompt, model_response) VALUES (?, ?)', (input_text, generated_text))
	conn.commit()
	conn.close()

	return jsonify({'response': generated_text})

@app.route("/info", methods=["GET"])
def get_info():
	return jsonify({
		'max_length': _max_length,
		'num_return_sequences': _num_return_sequences,
		'temperature' : _temperature,
		'top_p' : _top_p,
		'repetition_penalty' : _repetition_penalty,
		'do_sample': _do_sample
		})

@app.route('/history', methods=['GET'])
def get_history():
	results = []

	conn = sqlite3.connect('history.db', check_same_thread=False)
	cursor = conn.cursor()
	cursor.execute('SELECT * FROM messages')
	results = cursor.fetchall()
	conn.close()
	return jsonify(results)

if __name__ == '__main__':
	app.run(debug=True, host="0.0.0.0", port=5000)
