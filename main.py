import pandas as pd
import os, random
from sklearn.impute import SimpleImputer
import numpy as np
# from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from xgboost import XGBClassifier
from tqdm import tqdm
import warnings
import matplotlib.pyplot as plt
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.feature_selection import chi2
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import RandomizedSearchCV

np.seterr(all="ignore")
warnings.filterwarnings('ignore')


def enrich_data(df):
    if 'SepsisLabel' in df:
        df.drop(['SepsisLabel'], axis=1, inplace=True)
    df.reset_index(inplace=True)
    n = df.shape[0]

    clinical = df.columns[1:35]
    demographics = df.columns[34:41]
    sofa = []
    features = []

    for col1 in df.columns[1:]:
        features.append(np.nanmean(df[col1]))
        features.append(np.nanmin(df[col1]))
        features.append(np.nanmax(df[col1]))
        features.append(np.nanstd(df[col1]))
        features.append(df[col1].iloc[-1])
        features.append(df[col1].isnull().sum() / len(df))
        if col1 in clinical:
            features.append(df[col1].iloc[-1] - df[col1].iloc[0])

        if col1 in clinical:
            dv = df[col1].iloc[-1] - df[col1].iloc[0]
            features.append(dv)
            v = df[col1].iloc[-1]
            if col1 == 'Platelets':
                if v <= 40:
                    sofa.append(4)
                elif v <= 50:
                    sofa.append(3)
                elif v <= 100:
                    sofa.append(2)
                elif v <= 150:
                    sofa.append(1)
                else:
                    sofa.append(0)

            if col1 == 'Bilirubin_total':
                if v >= 11.9:
                    sofa.append(4)
                elif v >= 5.9:
                    sofa.append(3)
                elif v >= 1.9:
                    sofa.append(2)
                elif v >= 1.2:
                    sofa.append(1)
                else:
                    sofa.append(0)

            if col1 == 'Creatinine':
                if v >= 4.9:
                    sofa.append(4)
                elif v >= 3.4:
                    sofa.append(3)
                elif v >= 1.9:
                    sofa.append(2)
                elif v >= 1.2:
                    sofa.append(1)
                else:
                    sofa.append(0)

            if col1 == 'MAP':
                if v < 70:
                    sofa.append(1)
                else:
                    sofa.append(0)

    features += sofa
    features.append(sum(sofa))

    return features


def preprocess(path, mode):
    folder = path
    features_all_patient = []
    all_pateint = {}
    labels = []
    # scaler = preprocessing.MinMaxScaler()
    for filename in tqdm(os.listdir(folder)):
        patient_id = os.path.splitext(filename)[0]
        patient = pd.read_csv(os.path.join(folder, filename), delimiter="|")
        all_pateint[patient_id] = 1
        try:
            inx = patient[patient['SepsisLabel'] == 1].index.tolist()[0]
            new_patient = patient.iloc[:(inx + 1)]

            # print(patient['Unit2'])
        except:
            new_patient = patient
            all_pateint[patient_id] = 0

        # fit and transform the data for each column separately
        # for col in new_patient.columns:
        #     new_patient[[col]] = scaler.fit_transform(new_patient[[col]])
        features = enrich_data(new_patient)
        labels.append(all_pateint[patient_id])
        features_all_patient.append(features)
    df_x = pd.DataFrame(features_all_patient)

    # x = df_x.values  # returns a numpy array
    # standard_scaler = preprocessing.StandardScaler()
    # x_scaled = standard_scaler.fit_transform(x)
    # df = pd.DataFrame(x_scaled)
    return df_x, pd.Series(labels), list(all_pateint.keys())


def hp_tuning(X_train, y_train):
    from scipy.stats import uniform

    uni_dist = uniform()
    distributions = dict(n_estimators=list(range(15, 30)),
                         max_depth=list(range(5, 20)),
                         reg_alpha=uniform(),
                         reg_lambda=uniform(),
                         learning_rate=uniform())

    xgb_model = XGBClassifier(missing=np.nan)
    hp = RandomizedSearchCV(xgb_model, distributions, verbose=3)
    search = hp.fit(X_train, y_train)
    print(search.best_params_)
    return xgb_model


def th_tuning(X, y):
    pass


def evaluate(model, X_train, y_train,
             X_test, y_test, model_name):
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    test_prob = model.predict_proba(X_test)

    train_score, test_score = f1_score(y_train, train_pred), f1_score(y_test, test_pred)
    print(f'Train f1 score for model {model_name} is  {train_score}')
    print(f'Test f1 score for model {model_name} is {test_score}')

    res_train = pd.DataFrame.from_dict({'metric': ['F1 score', 'Accuracy', 'Precision', 'Recall'],
                                        'value': [f1_score(y_train, train_pred),
                                                  accuracy_score(y_train, train_pred),
                                                  precision_score(y_train, train_pred),
                                                  recall_score(y_train, train_pred)]})

    res_test = pd.DataFrame.from_dict({'metric': ['F1 score', 'Accuracy', 'Precision', 'Recall'],
                                       'value': [f1_score(y_test, test_pred),
                                                 accuracy_score(y_test, test_pred),
                                                 precision_score(y_test, test_pred),
                                                 recall_score(y_test, test_pred)]})

    pd.DataFrame.to_csv(res_train, model_name + '_train_scores.csv', index=False)
    pd.DataFrame.to_csv(res_test, model_name + '_test_scores.csv', index=False)

    return res_train, res_test


def get_dem(folder):
    res = []
    ids = []
    for filename in tqdm(os.listdir(folder)):
        patient = pd.read_csv(os.path.join(folder, filename), delimiter="|")
        demographics = patient.columns[34:41]
        data = patient[demographics].iloc[0].values
        ids.append(os.path.splitext(filename)[0])
        res.append(data)

    df = pd.DataFrame(res, columns=demographics)
    df['patient_id'] = ids
    pd.DataFrame.to_csv(df, 'test_dem.csv', index=False)


def sub_group(X, y, model, model_name,
              col, masks, names):

    data = []
    for m, n in zip(masks, names):
        Xi = X[m]
        yi = y[m]

        test_pred = model.predict(Xi)
        # print(f'f1 score for {n} is {f1_score(yi, test_pred)}')
        data.append([n,
                     f1_score(yi, test_pred),
                     accuracy_score(yi, test_pred),
                     precision_score(yi, test_pred),
                     recall_score(yi, test_pred)])

    data = pd.DataFrame(data, columns=['Group', 'F1 score', 'Accuracy', 'Precision', 'Recall'])
    pd.DataFrame.to_csv(data, model_name + '_' + col + '_scores.csv', index=False)


def sub_groups_metrics(X, y, model, model_name):
    #get_dem('data/test')
    dem = pd.read_csv('test_dem.csv')

    # Gender
    gender_mask = [dem['Gender'] == i for i in [0, 1]]
    genders = ['Women', 'Men']
    sub_group(X, y, model, model_name, 'Gender', gender_mask, genders)

    # Age
    age_mask = [(dem['Age'] > th) & (dem['Age'] < th + 20) for th in [0, 20, 40, 60, 80]]
    ages = [f'Ages {th} to {th + 20}' for th in [0, 20, 40, 60, 80]]
    sub_group(X, y, model, model_name, 'Ages', age_mask, ages)


def main():
    X_train, y_train, _ = preprocess(r"data/train", 'train')
    X_test, y_test, test_ids = preprocess(r"data/test", 'test')


    """ XGboost """
    xgb_model = XGBClassifier()
    xgb_model.load_model('model5.json')
    # xgb_model = hp_tuning(X_train, y_train)
    print("fitting Xgboost: ")
    xgb_model.fit(X_train, y_train)
    print("evaluating Xgboost: ")
    evaluate(xgb_model, X_train, y_train, X_test, y_test, 'XGBoost')
    sub_groups_metrics(X_test, y_test, xgb_model, 'XGBoost')

    # Random Forrest
    # reg = SelectKBest(mutual_info_regression, k=60).fit(X_train.fillna(-1), y_train)
    # X_new_train = reg.transform(X_train.fillna(-1))
    # X_new_test = reg.transform(X_test.fillna(-1))
    clf_RF = RandomForestClassifier(max_depth=10, random_state=0)
    print("fitting Random Forest: ")
    clf_RF.fit(X_train.fillna(-1), y_train)
    print("evaluating Random Forrest: ")
    evaluate(clf_RF, X_train.fillna(-1), y_train, X_test.fillna(-1), y_test, 'Random Forest')
    sub_groups_metrics(X_test.fillna(-1), y_test, clf_RF, 'Random Forest')

    # Logistic Regression
    clf_LR = LogisticRegression(random_state=0)
    print("fitting Logistic Regression: ")
    clf_LR.fit(X_train.fillna(-1), y_train)
    print("evaluating Logistic Regression: ")
    evaluate(clf_LR, X_train.fillna(-1), y_train, X_test.fillna(-1), y_test, ' Logistic Regression')
    sub_groups_metrics(X_test.fillna(-1), y_test, clf_LR, 'Logistic Regression')



if __name__ == '__main__':
    main()
