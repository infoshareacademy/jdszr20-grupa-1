from flask import Flask, request, jsonify, render_template
from xgboost import XGBClassifier
import joblib


app = Flask(__name__, template_folder='templates')
@app.route('/', methods=['GET'])
def home():
    return "Welcome to the fake test API!"

@app.route('/predict')
def predict_form():
    return render_template('my-form.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     data = request.get_json()
#     text = data.get('text', '')
#     p_text = help_predict(text)
#     return jsonify({'prediction': p_text})
@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']
    p_text = help_predict(text)
    return p_text

def my_form_post():
    text = request.form['text']
    processed_text = text.upper()
    return processed_text

def help_predict(text):
    with open('text_model.pkl', 'rb') as model_file:
        model = joblib.load(model_file)
    with open('text_vectorizer.pkl', 'rb') as vec_file:
        vectorizer = joblib.load(vec_file)
    v = vectorizer.transform([text]).toarray()
    p = model.predict(v)
    p_text = "Fake :(" if p[0] == 1 else "True!"
    return f"Prediction, is article fake news: {p_text}"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=4001, debug=True)
