import numpy as np
import time
from flask import jsonify
import json
from flask_cors import CORS
from flask import request
from flask import Flask
from sklearn.pipeline import Pipeline
from model import (load_dataset, preprocessing, train_test_split, model_testing,
                   model_training, load_model, save_model, feature_extraction, predict)

dataset_path = 'Dataset/Customer_data.csv'

app = Flask(__name__)
CORS(app)


@app.route('/', methods=['GET'])
def index():
    time.sleep(10)
    return 'Hello World!'


@app.route('/train', methods=['GET', 'POST'])
def train():
    data = load_dataset(dataset_path)
    print('Step1: Dataset is loaded successfully!')

    preprocessed_data = preprocessing(data)
    print('Step2: Data preprocessing done successfully!')

    train, test = train_test_split(preprocessed_data)
    print('Step3: Data splitted into train and test successfully!')

    train_X, train_Y, test_X, test_Y, vectorizer = feature_extraction(
        train, test)

    trained_model = model_training(train_X, train_Y)
    print('Step4: Model trained successfully successfully!')

    accuracy = model_testing(test_X, test_Y, trained_model)

    vec_classifier = Pipeline(
        [('vectorizer', vectorizer), ('classifier', trained_model)])

    save_model(vec_classifier)
    print('Step5: Model is deployed successfully')

    response = {'success': True,
                'message': 'Model deployed', 'accuracy': accuracy}
    return response


@app.route('/predict', methods=['POST'])
def store():
    model = load_model('saved_models/model.pkl')
    data = request.get_json()
    input = list(data['Input'])
    response = predict(input, model)
    response = response.tolist()
    response = {
        "result": response
    }
    return jsonify(response)


if __name__ == '__main__':
    app.run(debug=True, port=9000)
