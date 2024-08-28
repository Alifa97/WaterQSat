#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import configparser
import json
import ast

config_reader = configparser.ConfigParser()
config_reader.read('configurationWaterQSat.ini')

if config_reader.getboolean('RUN','Feature_Importance'):
    
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.inspection import permutation_importance
    import xgboost as xgb
    from xgboost import XGBRegressor
    import numpy as np
    import string
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    import matplotlib.pyplot as plt
    import seaborn as sns



    modelname = config_reader['Feature_engineering']['modelname']

    dataset_path = config_reader['Feature_engineering']['DatasetCSV']
    dataset_path = fr"{dataset_path}"
    dataset = pd.read_csv(dataset_path)

    dataset["Sat_date"] = pd.to_datetime(dataset["Sat_date"],format="%Y-%m-%d")
    dataset["Date"] = pd.to_datetime(dataset["Date"],format="%Y-%m-%d")


    
    list_bands = json.loads(config_reader['Feature_engineering']['bandsusedML'])
    list_meteo = json.loads(config_reader['Feature_engineering']['meteousedML'])
    list_target = json.loads(config_reader['Feature_engineering']['Target'])
    list_time = ['Month_sin','Month_cos','Delay']

    features = list_bands + list_meteo + list_target + list_time


    if config_reader['Feature_engineering']['Mission_used'] == 'Sentinel-2':
        for index, row in dataset.iterrows():
            if row["Sat_date"] > pd.Timestamp("2022-01-25"):
                dataset.loc[index,list_bands] -= 1000  #the offset applied in new updates of Sen2 dataset to avoid negative values which is subtracted here.
                ##add the link
        equation = lambda x: round((x * 0.0001), 9)  #rescaling the reflectance values which is stated here https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED#bands
        dataset[list_bands] = dataset[list_bands].applymap(equation)

    elif config_reader['Feature_engineering']['Mission_used'] == 'Landsat-8':
        equation = lambda x: round((x * 2.75e-05) - 0.2, 9)
        dataset[list_bands] = dataset[list_bands].applymap(equation)


    dataset["Delay"] = (dataset["Sat_date"] - dataset["Date"]).dt.days
    dataset["Month"] = (dataset["Date"]).dt.month

    # Convert the 'Month' column to float
    dataset['Month'] = dataset['Month'].astype(float)

    # Replace the 'Month' column with its sine and cosine
    dataset['Month_sin'] = np.sin(2 * np.pi * dataset['Month'] / 12)
    dataset['Month_cos'] = np.cos(2 * np.pi * dataset['Month'] / 12)


    # Drop the original 'Month' column
    dataset = dataset.drop('Month', axis=1)
    dataset = dataset.dropna(subset=features,inplace=False)
    dataset=dataset.sort_values(by=['Sat_date', list_target[0]],inplace=False)
    dataset = dataset.reset_index(drop=True)

    dataset_column_names = dataset.columns.tolist() 

    ###################Two bands indices

    Target_variable = list_target[0]
    df = dataset.copy()

    bands = list_bands
    equations = [("A/B", lambda A, B: A / B),
                 ("ln(A)", lambda A: np.log(A)),  # <-- changed here
                 ("(A+B) / (A-B)", lambda A, B: (A + B) / (A - B)),
                 ("(A-B) / (A+B)", lambda A, B: (A - B) / (A + B)),
                 ("A-B", lambda A, B: A - B),
                 ("A+B", lambda A, B: A + B),
                 ("A*B", lambda A, B: A * B),
                 ("A**2", lambda A: A ** 2),
                 ("1 / sqrt(A+B)", lambda A, B: 1 / np.sqrt(A + B)),
                 ("1 / sqrt(A-B)", lambda A, B: 1 / np.sqrt(A - B)),
                 ("ln(A / B)", lambda A, B: np.log(A / B))]

    df_to_concat = []
    for eq_name, eq in equations:
        column_eq = []
        for band1 in bands:
            for band2 in bands:
                if "A" in eq_name and "B" in eq_name:  # if the equation requires two bands
                    newcolumn = f"{eq_name}({band1}, {band2})"
                    column_eq.append(pd.DataFrame({newcolumn: eq(df[band1],df[band2])}))
                else:  
                    newcolumn = f"{eq_name}({band1})"
                    if newcolumn in [df.columns for df in column_eq]:  # if the column already exists
                        continue
                    column_eq.append(pd.DataFrame({newcolumn: eq(df[band1])}))       
        ChlA_df = pd.concat(column_eq,axis=1)
        df_to_concat.append(ChlA_df)

    df = pd.concat([df] + df_to_concat,axis = 1)
    nan_column = df.columns[df.isnull().any()]
    inf_column = df.select_dtypes(include=[np.number]).columns[np.isinf(df.select_dtypes(include=[np.number])).any()]
    dropcol = set(nan_column) | set(inf_column)
    df = df.drop(columns=dropcol)
    excluded_columns = dataset.columns.tolist()

    X = df[[col for col in df.columns if col not in excluded_columns + [Target_variable]]]
    y = df[Target_variable]

    Value_ranges = pd.cut(df[list_target[0]], bins=int(config_reader['Feature_engineering']['Featureimportance_bin']), labels=False)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=float(config_reader['Feature_engineering']['Featureimportance_testsize']), random_state=int(config_reader['Feature_engineering']['Featureimportance_randomstate']), stratify=Value_ranges)


    xgb_params = {
        'n_estimators': int(config_reader['Feature_engineering']['Featureimportance_XGB_N_estimators']),
        'max_depth': int(config_reader['Feature_engineering']['Featureimportance_XGB_max_depth']),
        'learning_rate': float(config_reader['Feature_engineering']['Featureimportance_XGB_learning_rate'])
    }

    xgb_model = XGBRegressor(objective='reg:squarederror', random_state=int(config_reader['Feature_engineering']['Featureimportance_randomstate']), **xgb_params)
    xgb_model.fit(X_train,y_train)

    feature_importance_xgb = xgb_model.feature_importances_
    threshold_xgb = float(config_reader['Feature_engineering']['Featureimportance_Tresholad_score'])
    indices_importantfeatures = [ i for i, importance in enumerate(feature_importance_xgb) if importance > threshold_xgb]
    importantfeature_xgb = X.columns[indices_importantfeatures]
    df2band = df[importantfeature_xgb]
    importancescore_xgb = feature_importance_xgb[indices_importantfeatures]
    print(f"The important indices with two bands are as following:")
    for feature, importance in zip(importantfeature_xgb,importancescore_xgb):
        print(f"{feature}: {importance}")

    ###################Three bands indices


    Target_variable = list_target[0]
    df = dataset.copy()
    equations_three_bands = [
        ("(A * B) / C", lambda A, B, C: (A * B) / C),
        ("((1/A) - (1/B)) * C", lambda A, B, C: (((1/A) - (1/B)) * C)),
        ("(A + B + C)", lambda A, B, C: (A + B + C)),
        ("A / (B + C)", lambda A, B, C: A / (B + C)),
        ("(A + B) / C", lambda A, B, C: (A + B) / C),
        ("(A - B) / (A + B + C)", lambda A, B, C: (A - B) / (A + B + C)),
        ("(A + B + C) / (A - B)", lambda A, B, C: (A + B + C) / (A - B)),
        ("(A - B) / (A + B) / C", lambda A, B, C: (A - B) / (A + B) / C),
        ("(A + B) / (A - C)", lambda A, B, C: (A + B) / (A - C)),
        ("((A - B) / (A + B)) / ((A - C) / (A + C))", lambda A, B, C: ((A - B) / (A + B)) / ((A - C) / (A + C))),
    ]

    # List of bands
    bands_three = list_bands


    new_columns_dfs = []
    # Apply equations and add columns to the DataFrames
    for equation_name, equation_func in equations_three_bands:

        for band1 in bands_three:
            for band2 in bands_three:
                for band3 in bands_three:
                    # Create a new DataFrame with the equation result
                    new_column_name = f"{equation_name}({band1}, {band2}, {band3})"
                    new_df = pd.DataFrame({new_column_name: equation_func(df[band1], df[band2], df[band3])})
                    new_columns_dfs.append(new_df)

    df = pd.concat([df] + new_columns_dfs, axis=1)
    nan_columns = df.columns[df.isnull().any()]
    inf_columns = df.select_dtypes(include=[np.number]).columns[np.isinf(df.select_dtypes(include=[np.number])).any()]
    columns_to_drop = set(nan_columns) | set(inf_columns)
    df = df.drop(columns=columns_to_drop)

    excluded_columns = dataset.columns.tolist()


    X = df[[col for col in df.columns if col not in excluded_columns + [Target_variable]]]
    y = df[Target_variable]

    # Split data into train and test sets
    Value_ranges = pd.cut(df[list_target[0]], bins=int(config_reader['Feature_engineering']['Featureimportance_bin']), labels=False)

    # Split the data into training and testing sets, ensuring stratified sampling based on ChlA ranges
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=float(config_reader['Feature_engineering']['Featureimportance_testsize']), random_state=int(config_reader['Feature_engineering']['Featureimportance_randomstate']), stratify=Value_ranges)


    xgb_params = {
        'n_estimators': int(config_reader['Feature_engineering']['Featureimportance_XGB_N_estimators']),
        'max_depth': int(config_reader['Feature_engineering']['Featureimportance_XGB_max_depth']),
        'learning_rate': float(config_reader['Feature_engineering']['Featureimportance_XGB_learning_rate'])
    }

    xgb_model = XGBRegressor(objective='reg:squarederror', random_state=int(config_reader['Feature_engineering']['Featureimportance_randomstate']), **xgb_params)
    xgb_model.fit(X_train,y_train)

    feature_importance_xgb = xgb_model.feature_importances_

    threshold_xgb = float(config_reader['Feature_engineering']['Featureimportance_Tresholad_score'])


    important_indices_xgb = [i for i, importance in enumerate(feature_importance_xgb) if importance > threshold_xgb]
    important_features_xgb = X.columns[important_indices_xgb]
    important_importances_xgb = feature_importance_xgb[important_indices_xgb]
    df3band = df[important_features_xgb]
    print(f"The important indices with three bands are as following:")
    for feature, importance in zip(important_features_xgb, important_importances_xgb):
        print(f"{feature}: {importance}")

    ###################Four bands indices


    Target_variable = list_target[0]
    df = dataset.copy()

    # List of equations for up to 4 bands
    equations_four_bands = [
        ("A + B + C + D", lambda A, B, C, D: A + B + C + D),
        ("A / (B + C + D)", lambda A, B, C, D: A / (B + C + D)),
        ("(A + B) / (C + D)", lambda A, B, C, D: (A + B) / (C + D)),
        ("(A - B) / (C + D)", lambda A, B, C, D: (A - B) / (C + D)),
        ("((A - B) / (A + B)) / ((C - D) / (C + D))", lambda A, B, C, D: ((A - B) / (A + B)) / ((C - D) / (C + D))),
    ]

    # List of bands
    bands_four = list_bands

    new_columns_dfs = []
    for equation_name, equation_func in equations_four_bands:
        for band1 in bands_four:
            for band2 in bands_four:
                for band3 in bands_four:
                    for band4 in bands_four:
                        # Create a new DataFrame with the equation result
                        new_column_name = f"{equation_name}({band1}, {band2}, {band3}, {band4})"
                        new_df = pd.DataFrame({new_column_name: equation_func(df[band1], df[band2], df[band3], df[band4])})
                        new_columns_dfs.append(new_df)



    # Concatenate all DataFrames in the list along axis=1
    df = pd.concat([df] + new_columns_dfs, axis=1)

    nan_columns = df.columns[df.isnull().any()]
    inf_columns = df.select_dtypes(include=[np.number]).columns[np.isinf(df.select_dtypes(include=[np.number])).any()]
    columns_to_drop = set(nan_columns) | set(inf_columns)
    df = df.drop(columns=columns_to_drop)
    excluded_columns = dataset.columns.tolist()


    X = df[[col for col in df.columns if col not in excluded_columns + [Target_variable]]]
    y = df[Target_variable]


    # Split data into train and test sets
    Value_ranges = pd.cut(df[list_target[0]], bins=int(config_reader['Feature_engineering']['Featureimportance_bin']), labels=False)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=float(config_reader['Feature_engineering']['Featureimportance_testsize']), random_state=int(config_reader['Feature_engineering']['Featureimportance_randomstate']), stratify=Value_ranges)

    xgb_params = {
        'n_estimators': int(config_reader['Feature_engineering']['Featureimportance_XGB_N_estimators']),
        'max_depth': int(config_reader['Feature_engineering']['Featureimportance_XGB_max_depth']),
        'learning_rate': float(config_reader['Feature_engineering']['Featureimportance_XGB_learning_rate'])
    }

    xgb_model = XGBRegressor(objective='reg:squarederror', random_state=int(config_reader['Feature_engineering']['Featureimportance_randomstate']), **xgb_params)
    xgb_model.fit(X_train,y_train)

    feature_importance_xgb = xgb_model.feature_importances_
    threshold_xgb = float(config_reader['Feature_engineering']['Featureimportance_Tresholad_score'])
    important_indices_xgb = [i for i, importance in enumerate(feature_importance_xgb) if importance > threshold_xgb]
    important_features_xgb = X.columns[important_indices_xgb]
    important_importances_xgb = feature_importance_xgb[important_indices_xgb]
    df4band = df[important_features_xgb]
    print(f"The important indices with four bands are as following:")
    for feature, importance in zip(important_features_xgb, important_importances_xgb):
        print(f"{feature}: {importance}")

    indices = pd.concat([df2band, df3band, df4band], axis=1)
    dataset = pd.concat([dataset,indices],axis=1)

    
    ###################Literature indices only for ChlA

    if config_reader.getboolean('Feature_engineering','MLforChlA'):
        dataset['TBDOI'] = dataset['B06'] * ((1 / dataset['B04']) - (1 / dataset['B05']))#*(Towards a unified approach for remote estimation of chlorophyll-a in both terrestrial vegetation and turbid productive waters)
        dataset['QB4'] = (dataset['B06']-dataset['B02'])/(dataset['B03']+dataset['B05'])#*(Validación de algoritmos para la estimación de la Clorofila-a con Sentinel-2 en la Albufera de Valencia translated paper of Validacionalgoritmosclorofila)
        dataset['ND4I'] = (dataset['B03']-dataset['B05'])/(dataset['B03']+dataset['B05'])#*ND4I
        dataset['ND4II'] = (dataset['B05']-dataset['B04'])/(dataset['B05']+dataset['B04'])#*ND4II(Evaluation of Sentinel-2 and Landsat 8 Images for Estimating Chlorophyll-a Concentrations in Lake Chad, Africa)
        dataset['GR'] = (dataset['B03']/dataset['B04'])  #*canceled for writing cuz low correlation(Selecting the Best Band Ratio to Estimate Chlorophyll-a Concentration in a Tropical Freshwater Lake Using Sentinel 2A Images from a Case Study of Lake Ba Be (Northern Vietnam))
        dataset['TBDOII'] = dataset['B8A'] * ((1 / dataset['B04']) - (1 / dataset['B05']))#*TBDOII(Ref folder ChlA index TBDO2)
        dataset['NIRR'] = (dataset['B05']/dataset['B04'])#*canceled for writing cuz low correlation(Selecting
        dataset = dataset.dropna(axis=1)


    ###################Final Top features

    columns_to_exclude = json.loads(config_reader['Feature_engineering']['Featureimportance_Excludefeatures'])
    X = dataset[[col for col in dataset.columns if col not in excluded_columns + [Target_variable]]]
    y = dataset[list_target[0]]  
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=float(config_reader['Feature_engineering']['Featureimportance_testsize']), random_state=int(config_reader['Feature_engineering']['Featureimportance_randomstate']))
    xgb_params = {
        'n_estimators': int(config_reader['Feature_engineering']['Featureimportance_XGB_N_estimators']),
        'max_depth': int(config_reader['Feature_engineering']['Featureimportance_XGB_max_depth']),
        'learning_rate': float(config_reader['Feature_engineering']['Featureimportance_XGB_learning_rate'])
    }
    xgb_model = XGBRegressor(objective='reg:squarederror', random_state=int(config_reader['Feature_engineering']['Featureimportance_randomstate']), **xgb_params)
    xgb_model.fit(X_train,y_train)
    feature_importances = xgb_model.feature_importances_

    # Get the indices of the top features
    top_feature_indices = np.argsort(feature_importances)[::-1][:int(config_reader['Feature_engineering']['Top_features'])]
    # Get the names and importance scores
    top_features = X.columns[top_feature_indices]
    top_features_importances = feature_importances[top_feature_indices]

    print("Top Important Features:")
    for feature, importance in zip(top_features, top_features_importances):
        print(f"{feature}: {importance}")

    plt.figure(figsize=(10, 6), facecolor='white')
    plt.barh(top_features, top_features_importances, color='blue')
    plt.xlabel('Importance Score')
    plt.ylabel('Features')
    plt.title(fr'Features Importance values for {modelname}')
    Savefeatureimportanceplot = config_reader['Feature_engineering']['Plot_saving_path']
    plt.savefig(fr'{Savefeatureimportanceplot}\{modelname}_featureimportance.png', bbox_inches='tight', pad_inches=0.5, facecolor='w')
    plt.show()


    ###################Correlation analysis
    correlation_matrix = dataset[top_features].corr()
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    np.fill_diagonal(correlation_matrix.values, 1.0)
    plt.figure(figsize=(18, 13), facecolor='white')

    heatmap = sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', fmt=".2f", linewidths=3, mask=~mask, cbar=True, vmin=-1, vmax=1, cbar_kws={"shrink": 0.75, "pad": 0.15, "location": "left"})
    cbar = heatmap.collections[0].colorbar
    cbar.ax.tick_params(labelsize=14)
    cbar.set_label('Correlation', fontsize=14, labelpad=15)
    plt.tick_params(axis='x', top=True, labeltop=True, bottom=False, labelbottom=False, rotation=90, labelsize=12)
    xticks_positions = np.arange(len(correlation_matrix.columns))
    xticks_labels = correlation_matrix.columns
    heatmap.set_xticks(xticks_positions + 0.5)
    heatmap.set_xticklabels(xticks_labels, fontsize=16)
    plt.tick_params(axis='y', right=True, labelright=True, left=False, labelleft=False, rotation=0, labelsize=12)
    yticks_positions = np.arange(len(correlation_matrix.index))
    yticks_labels = correlation_matrix.index
    heatmap.set_yticks(yticks_positions + 0.5)
    heatmap.set_yticklabels(yticks_labels, fontsize=16)
    plt.title(fr'Correlation Matrix of Indices for {modelname}', fontsize=16)

    plt.savefig(fr'{Savefeatureimportanceplot}\{modelname}_corrmatrix.png', bbox_inches='tight', pad_inches=0.5, facecolor='w')

    plt.show()

