from main import *
import numpy as np
import pickle
import pandas as pd
import argparse
from xgboost import XGBClassifier

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process input')
    parser.add_argument('input_folder', type=str, help='Input folder path, containing images')
    args = parser.parse_args()
    path = args.input_folder

    X_test, y_test, ids = preprocess(path, 'comp')

    index = [int(i[i.find('patient_') + len('patient_'):]) for i in ids]

    model = XGBClassifier()
    model.load_model('comp.json')

    pred = model.predict(X_test)

    res = pd.DataFrame.from_dict({'id': ids,
                                  'prediction': pred,
                                  'index': index})
    res = res.sort_values('index').drop(['index'], axis=1)
    pd.DataFrame.to_csv(res, 'prediction.csv', index=False)

