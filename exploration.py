import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from tqdm import tqdm
import os, random
from typing import Dict, Tuple
from sklearn.decomposition import PCA 
from sklearn.impute import SimpleImputer


# The Data
# The data files are stored in psv files

def get_patient(file_name: str) -> pd.DataFrame:
    patient_data = pd.read_csv(file_name, delimiter='|')
    return patient_data

def get_patients_dicts(folder_name: str) -> Tuple[Dict[int, pd.DataFrame], Dict[int, pd.DataFrame]]:
    """
    Args:
        folder_name: The path of the folder that holds the psv files of the patients
    Returns:
        A dictionary of patient_number: pd.Dataframe
        Note: This function will eliminate any sick patients will less than 6 hours before the attack.
    """
    patient_dict_attack = dict()
    patient_dict_no_attack = dict()
    for dirpath, dirnames, filenames in os.walk(folder_name):
        for filename in tqdm(filenames):
            patient_id = os.path.splitext(filename)[0].split('_')[1]
            patient_df = get_patient(os.path.join(dirpath, filename))
            trimmed_patient_df, had_attack = trim_patient(patient_df)
            if len(trimmed_patient_df)==0:
                continue
            if had_attack:
                patient_dict_attack[int(patient_id)] = trimmed_patient_df
            else:
                patient_dict_no_attack[int(patient_id)] = trimmed_patient_df
    return patient_dict_attack, patient_dict_no_attack

def trim_patient(df: pd.DataFrame) -> Tuple[pd.DataFrame, bool]:
    """
    Trims the patient df to the first appearence of the spesis label (if there is one).
    Args:
        df: The patient dataframe
    Returns:
        A dataframe that goes up to and including the start of the attack.
        A Boolean value for wherther the patient suffered an attack.
    """
    had_attack = 1 if sum(df['SepsisLabel']) > 0 else 0
    if had_attack:
        attack_start = int(df['SepsisLabel'].idxmax())
        trimmed = df.iloc[:attack_start+1] # Take up-tuntil the attack.
        df_final = pd.DataFrame()
        if len(trimmed) >= 10:
            df_final = trimmed[-10:].reset_index() # Take only the last 9 hours before the attack. 
        return df_final, True
    df_final = df[:10]
    return df_final.reset_index(), False

def get_feature(feature: str, patient_dict: Dict) -> pd.DataFrame:
    """
    Creates a verticle dataframe of one feature for all patients.
    Args:
        feature: The wanted feature
        patient_dict: The collection of patients and all their data.
    Returns:
        A Dataframe of that feature of all patients. Each column is a patient and the columns are parallel hours.
    """
    df_out = pd.DataFrame()
    cols = []
    for patient, data in patient_dict.items():
        col = data[feature].copy()
        cols.append(col)
    df_out = pd.concat(cols, axis=1)
    return df_out

def get_averages(sick: Dict, healthy: Dict) -> pd.DataFrame:
    """
    Recieves the dictionaries of the healthy and sick patient dataframes
    and returns one dataframe with all patient averages with a label column
    """
    averages = []
    for patient, data in sick.items():
        patient_averages = data.drop('SepsisLabel', axis=1).mean()
        df = pd.DataFrame(patient_averages).T
        df['SepsisLabel'] = 1
        averages.append(df)
    for patient, data in healthy.items():
        patient_averages = data.mean()
        df = pd.DataFrame(patient_averages).T
        averages.append(df)
    df_out = pd.concat(averages)
    return df_out

def pca_patients(sick: Dict, healthy: Dict) -> pd.DataFrame:
    """
    PCA for all the data
    """
    imp = SimpleImputer(missing_values=np.nan, strategy='median')
    pca = PCA(n_components=3)

    data = get_averages(sick, healthy)
    data = data.drop('index', axis=1)
    data_dropped = data.dropna(axis=1, how='all') # Drop Features that are all Nan
    df_imputed = pd.DataFrame(imp.fit_transform(data_dropped), columns=list(data_dropped.columns))
    
    X = df_imputed.drop('SepsisLabel', axis=1).values
    y = df_imputed['SepsisLabel'].values

    # Standardize the feature columns
    X_std = (X - X.mean(axis=0)) / X.std(axis=0)
    X_std = np.nan_to_num(X_std, nan=0)

    pca.fit(X_std)
    X_pca = pca.transform(X_std)
    pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2', 'PC3'])
    pca_df['SepsisLabel'] = y
    return pca_df

def get_catagory_distributions(sick: Dict, healthy: Dict):
    sick_male = 0
    healthy_male = 0
    male = 0
    sick_female = 0
    healthy_female = 0 
    female = 0
    sick_ages = []
    healthy_ages = []
    for data in sick.values():
        gender = data['Gender'].loc[0]
        age = data['Age'].loc[0]
        sick_ages.append(age)
        if gender > 0:
            sick_male+=1
            male+=1
        else:
            sick_female+=1
            female+=1
    for data in healthy.values():
        gender = data['Gender'].loc[0]
        age = data['Age']
        healthy_ages.append(age)
        if gender > 0:
            healthy_male+=1
            male+=1
        else:
            healthy_female+=1
            female+=1
    gender_dict = {"Males": [sick_male, healthy_male], "Females":[sick_female, healthy_female]}
    sns.histplot(sick_ages,label="Attack")
    sns.histplot(healthy_ages, label="No Attack")
    plt.show()
    df = pd.DataFrame.from_dict(gender_dict, orient='index', columns=['Attack', 'No Attack'])
    df['gender'] = df.index
    df_melted = pd.melt(df, id_vars=['gender'], value_vars=['Attack', 'No Attack'], var_name='Label', value_name='count')
    sns.barplot(x='gender', y='count', hue='Label', data=df_melted)
    plt.title('Number of Attacks males and females')
    plt.show()
    
patient_dict_attack, patient_dict_no_attack = get_patients_dicts("/home/student/hw1/data")

some_patient = list(patient_dict_attack.values())[0]
columns = some_patient.columns

def make_line_plots(columns, patient_dict_attack, patient_dict_no_attack):
    for feature in tqdm(columns):
        sick = get_feature(feature, patient_dict_attack)
        healthy = get_feature(feature, patient_dict_no_attack)
        sick_mean = sick.mean(axis=1)
        healthy_mean = healthy.mean(axis=1)
        sns.lineplot(x=healthy.index, y=healthy_mean, label="No Attack")
        sns.lineplot(x=sick.index, y=sick_mean, label="Attack").set(title=feature)
        plt.xlabel("Hours")
        plt.ylabel(f"Average {feature} Measure")
        filename = "/home/student/hw1/data_visualisation/plots_10_hours/"+feature+".png"
        plt.savefig(filename)
        plt.clf()
    return


def do_pca(sick: Dict, healthy: Dict) -> None:
    """
    Performs PCA reduction to the data and plots the scatter plot.
    """
    reduced_data = pca_patients(sick, healthy)
    reduced_data_shifted = reduced_data + 5
    reduced_data_clipped = reduced_data_shifted.clip(lower=0)
    attack = reduced_data_shifted[reduced_data_shifted['SepsisLabel'] > 5]
    no_attack = reduced_data_shifted[reduced_data_shifted['SepsisLabel'] < 6]
    print(reduced_data)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(attack['PC2'], attack['PC1'], attack['PC3'], c='blue', marker='o', label='Attack')
    ax.scatter(no_attack['PC2'], no_attack['PC1'], no_attack['PC3'], c='orange', marker='o',label='No Attack')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    plt.title('PCA of patient feature averages to n=3')
    plt.legend()
    filename = "/home/student/hw1/data_visualisation/plots_10_hours/PCA3_shifted_rot.png"
    plt.savefig(filename)
    plt.show()
    return


# get_catagory_distributions(patient_dict_attack, patient_dict_no_attack)
# make_line_plots(columns, patient_dict_attack, patient_dict_no_attack)
do_pca(patient_dict_attack, patient_dict_no_attack)

print(columns)