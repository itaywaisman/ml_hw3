import pandas as pd
from preprocessing.constants import bad_features, continous_features, pre_final_list, pcrs, others, final_features, selected_features
from preprocessing.transformations import split_data, features_data_types_pipeline, label_transformer
from preprocessing.imputer import Imputer
from preprocessing.outlier_detection import OutlierClipper
from preprocessing.normalizer import Normalizer
from preprocessing.feature_selection import select_features_filter, select_features_wrapper
from preprocessing.visualize import display_correlation_matrix, save_scatter_plots, plot_df_scatter
from sklearn.pipeline import Pipeline



def load_data(filename):
    # Load Dataset
    df = pd.read_csv(filename)

    return df
    
def split(df):
    df = df.drop(labels=bad_features, axis=1)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)
    
    return X_train, X_val, X_test, y_train, y_val, y_test
    
def prepare_data(X_train, X_val, X_test, y_train, y_val, y_test):
    
    patient_ids = X_train['PatientID'], X_val['PatientID'], X_test['PatientID']
    X_train, X_val, X_test = X_train.drop(labels=['PatientID'], axis=1), X_val.drop(labels=['PatientID'], axis=1), X_test.drop(labels=['PatientID'], axis=1)
    # Prepare Dataset
    data_preperation_pipelines = Pipeline([
        ('feature_types', features_data_types_pipeline),
        ('feature_imputation', Imputer()),
        ('outlier_clipping', OutlierClipper(features=continous_features)),
        ('normalization', Normalizer())
    ])
    data_preperation_pipelines.fit(X_train, y_train)
    label_transformer.fit(y_train)
    X_train_prepared, y_train_prepared = data_preperation_pipelines.transform(X_train), label_transformer.transform(
        y_train)
    X_validation_prepared, y_validation_prepared = data_preperation_pipelines.transform(
        X_val), label_transformer.transform(y_val)
    X_test_prepared, y_test_prepared = data_preperation_pipelines.transform(X_test), label_transformer.transform(y_test)

    X_train_prepared, X_validation_prepared, X_test_prepared = pd.concat([patient_ids[0], X_train_prepared], axis=1), pd.concat([patient_ids[1], X_validation_prepared], axis=1), pd.concat([patient_ids[2], X_test_prepared], axis=1)

    
    return X_train_prepared, X_validation_prepared, X_test_prepared,\
           y_train_prepared, y_validation_prepared, y_test_prepared


def save_selected_features(original_list, final_list):
    df = pd.DataFrame([1 if feature in final_list else 0 for feature in original_list], index=original_list)
    
    df.T.to_csv('selected_columns.csv')
    
def preprocess(X_train, X_val, X_test, y_train, y_val, y_test):
    
    X_train_prepared, X_validation_prepared, X_test_prepared,\
    y_train_prepared, y_validation_prepared, y_test_prepared = prepare_data(X_train, X_val, X_test, y_train, y_val, y_test)
    
    X_train_final, X_validation_final, X_test_final = X_train_prepared[final_features], X_validation_prepared[final_features], X_test_prepared[final_features]

    return X_train_final, X_validation_final, X_test_final, y_train_prepared, y_validation_prepared, y_test_prepared

    
