import pandas as pd
from preprocessing.constants import bad_features, continous_features, pre_final_list, pcrs, others, final_features, selected_features
from preprocessing.transformations import split_data, features_data_types_pipeline, label_transformer
from preprocessing.imputer import Imputer
from preprocessing.outlier_detection import OutlierClipper
from preprocessing.normalizer import Normalizer
from preprocessing.feature_selection import select_features_filter, select_features_wrapper
from preprocessing.visualize import display_correlation_matrix, save_scatter_plots, plot_df_scatter
from sklearn.pipeline import Pipeline

class DataPreprocessor:

    def __init__(self):
        self.data_preperation_pipelines = Pipeline([
            ('feature_types', features_data_types_pipeline),
            ('feature_imputation', Imputer()),
            ('outlier_clipping', OutlierClipper(features=continous_features)),
            ('normalization', Normalizer())
        ])

    def fit(self, X, y, **kwargs):

        X_tag = X.drop(labels=['PatientID', *bad_features], axis=1)

        self.data_preperation_pipelines.fit(X_tag, y)

        return self
    
    def transform(self, X, **kwargs):
        patient_ids = X['PatientID']
        X_tag = X.drop(labels=['PatientID', *bad_features], axis=1)

        X_transformed =  self.data_preperation_pipelines.transform(X_tag)

        X_transformed = pd.concat([patient_ids, X_transformed], axis=1)
        X_transformed = X_transformed[final_features]
        return X_transformed

    def fit_transform(self, X, y, **kwargs):
        self.fit(X, y, **kwargs)
        return self.transform(X)

def split(df):
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)
    
    return X_train, X_val, X_test, y_train, y_val, y_test