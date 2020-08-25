from model import (load_dataset, preprocessing, train_test_split, model_testing,
                   model_training, load_model, save_model, feature_extraction, predict, append_list_as_row)
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.pipeline import Pipeline

dataset_path = 'Dataset/Customer_data.csv'

try:
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

    loaded_model = load_model('saved_models/model.pkl')

    data = ["Aortic aneurysm", "AR", "SAS", "HBP", "hipertynson"]
    res = predict(data, loaded_model)
    print(res)

except Exception as e:
    print('Error occured in the main loop:  ', e)
