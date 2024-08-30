#!/usr/bin/env python
# coding: utf-8

# In[28]:


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
    list_Date = ['Sat_date','Date']
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
    X = dataset[[col for col in dataset.columns if col not in columns_to_exclude + [Target_variable]]]
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

    dataset_final = dataset[list_Date + ["ROI"]+ list_target + top_features.to_list()]
    
            
    finaldatasetcsvpath = config_reader['Feature_engineering']['Savecsvpath']
    finaldatasetcsvpath = fr"{finaldatasetcsvpath}"
    dataset_final.to_csv(finaldatasetcsvpath, index=False)
    print(f"File saved in {finaldatasetcsvpath}.")
    
    
    
    
    

if config_reader.getboolean('RUN','ML'):
    

    import pandas as pd
    import xgboost as xgb
    from sklearn.metrics import mean_absolute_error, mean_squared_error,r2_score
    from sklearn.model_selection import train_test_split, KFold
    from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
    import matplotlib.pyplot as plt
    from sklearn.feature_selection import RFE
    import joblib



    modelname = config_reader['ML']['modelname']

    dataset_path = config_reader['ML']['MLDatasetCSV']
    dataset_path = fr"{dataset_path}"
    dataset = pd.read_csv(dataset_path)

    dataset["Sat_date"] = pd.to_datetime(dataset["Sat_date"],format="%Y-%m-%d")
    dataset["Date"] = pd.to_datetime(dataset["Date"],format="%Y-%m-%d")

    testsize = float(config_reader['ML']['ML_testsize'])
    folds = int(config_reader['ML']['ML_crossfold_validation_folds'])
    classifiedbins = int(config_reader['ML']['ML_classification_bins'])
    Nfeature = int(config_reader['ML']['ML_number_features'])
    Randomstate = int(config_reader['ML']['Randomstate'])                

    list_target = json.loads(config_reader['ML']['Targetfeature'])



    X = dataset[json.loads(config_reader['ML']['Features_used'])]

    y = dataset[list_target[0]]





    dataset_ranges = pd.cut(dataset[list_target[0]], bins=classifiedbins,labels=False)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testsize, random_state=Randomstate)

    # Initialize RFE
    selector = RFE(xgb.XGBRegressor(objective='reg:squarederror'), n_features_to_select=Nfeature)

    # Fit RFE
    selector = selector.fit(X_train, y_train)

    # Get selected features
    selected_features = X.columns[selector.support_]


    print("testsize", testsize)
    print("Kfolds", folds)
    print("classified bins", classifiedbins)
    print("Number of features", Nfeature)

    print("Selected feature:", selected_features)


    # Split the data into training and testing sets, ensuring stratified sampling based on ChlA ranges
    dataset_ranges = pd.cut(dataset[list_target[0]], bins=classifiedbins,labels=False)
    X_train, X_test, y_train, y_test = train_test_split(X[selected_features], y, test_size=testsize, random_state=Randomstate, stratify=dataset_ranges)



    # function to optimize (based on R-squared)
    def objective(params):
        xgb_model = xgb.XGBRegressor(objective='reg:squarederror', **params)
        r2_list = []
        mae_list = []
        rmse_list = []

        kfold = KFold(n_splits=folds, shuffle=True, random_state=Randomstate)
        #kfold = KFold(n_splits=folds, shuffle=False)
        for train_index, test_index in kfold.split(X_train):
            X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[test_index]
            y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[test_index]

            # Train the model with the current hyperparameters on the training fold
            xgb_model.fit(X_train_fold, y_train_fold)

            # Evaluate the model on the validation fold
            y_pred_fold = xgb_model.predict(X_val_fold)

            # Calculate R-squared- MAE - RMSE for the validation fold
            r2 = r2_score(y_val_fold, y_pred_fold)
            r2_list.append(r2)
            mae = mean_absolute_error(y_val_fold, y_pred_fold)
            mae_list.append(mae)
            rmse = mean_squared_error(y_val_fold, y_pred_fold, squared=False)
            rmse_list.append(rmse)



        # Calculate averages
        avg_r2 = sum(r2_list) / len(r2_list)
        avg_mae = sum(mae_list) / len(mae_list)
        avg_rmse = sum(rmse_list) / len(rmse_list)


        if config_reader['ML']['Loss_optimization'] == 'r2':
            loss = -avg_r2
        elif config_reader['ML']['Loss_optimization'] == 'mae':
            loss = avg_mae
        elif config_reader['ML']['Loss_optimization'] == 'rmse':
            loss = avg_rmse
        else:
            raise ValueError(f"Invalid Loss_optimization value: {config_reader['ML']['Loss_optimization']}")

        return {'loss': loss, 'rmse': avg_rmse, 'mae':avg_mae, 'Rsquare':avg_r2, 'status': STATUS_OK}



    list_n_estimators = json.loads(config_reader['ML']['n_estimators'])
    list_max_depth = json.loads(config_reader['ML']['max_depth'])
    list_learning_rate = json.loads(config_reader['ML']['learning_rate'])
    list_alpha = json.loads(config_reader['ML']['alpha'])
    list_lambda = json.loads(config_reader['ML']['lambda'])
    list_gamma = json.loads(config_reader['ML']['gamma'])




    # hyperparameter search space
    space = {
        'n_estimators': hp.choice('n_estimators', range(list_n_estimators[0], list_n_estimators[1], list_n_estimators[2])),
        'max_depth': hp.choice('max_depth', range(list_max_depth[0], list_max_depth[1], list_max_depth[2])),
        'learning_rate': hp.uniform('learning_rate', list_learning_rate[0], list_learning_rate[1]),
        'alpha': hp.uniform('alpha', list_alpha[0], list_alpha[1]),  
        'lambda': hp.uniform('lambda', list_lambda[0], list_lambda[1]),
        'gamma': hp.uniform('gamma', list_gamma[0], list_gamma[1])
    }


    # keep track of optimization results
    trials = Trials()

    # Perform Bayesian Optimization to maximize R-squared by using fmin function
    best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=int(config_reader['ML']['Opt_iterations']), trials=trials)

    # Get the best hyperparameters by using the indices of them
    best_xgb_params = {
        'n_estimators': range(list_n_estimators[0], list_n_estimators[1], list_n_estimators[2])[best['n_estimators']],
        'max_depth': range(list_max_depth[0], list_max_depth[1], list_max_depth[2])[best['max_depth']],
        'learning_rate': best['learning_rate'],
        'alpha': best['alpha'],
        'lambda': best['lambda'],
        'gamma': best['gamma']
    }

    # Train the final model on the training set with the best hyperparameters
    final_xgb_model = xgb.XGBRegressor(objective='reg:squarederror', **best_xgb_params)
    final_xgb_model.fit(X_train, y_train)

    # Use the final model on the test set
    y_pred_test = final_xgb_model.predict(X_test)

    #time series plot


    # Calculate metrics on the test set
    r2_test = r2_score(y_test, y_pred_test)
    mae_test = mean_absolute_error(y_test, y_pred_test)
    rmse_test = mean_squared_error(y_test, y_pred_test, squared=False)

    # Print the results
    print("Best Hyperparameters:")
    print(best_xgb_params)



    print("\nTest Metrics:")
    print("Test R-squared:", r2_test)
    print("Test MAE:", mae_test)
    print("Test RMSE:", rmse_test)

    best_metrics = trials.best_trial['result']
    print("\nValidation Metrics for Best Hyperparameters:")
    print("Validation R-squared:", best_metrics["Rsquare"])
    print("Validation MAE:", best_metrics['mae'])
    print("Validation RMSE:", best_metrics['rmse'])


    plt.scatter(y_test, y_pred_test, alpha=0.5)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', linewidth=2)
    plt.xlabel("True Value")
    plt.ylabel("Predicted Value")
    plt.title(fr"{modelname}: True vs. Predicted")
    Savescatterplot = config_reader['ML']['Plot_saving_path']
    plt.savefig(fr'{Savescatterplot}\{modelname}.png', bbox_inches='tight', pad_inches=0.5, facecolor='w') 
    plt.show()

    Savemodel = config_reader['ML']['ML_saving_path']
    joblib.dump(final_xgb_model, fr'{Savemodel}\{modelname}.joblib')
    
    
    
    

