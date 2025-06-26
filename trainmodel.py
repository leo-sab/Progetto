
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
import sklearn.metrics as metrics 
from sklearn.preprocessing import LabelEncoder
import polars as pl
import joblib  
import numpy as np
from app import data

# set random seed for reproducibility and parameters
seed = 42
num_c = 500 # min number of prenotations for keeping a country
n_estimators = 100 # number of trees in the forest
test_size = 0.2 # cross validation test size


def preproc():
   
    df = data.drop("index","arrival_date","arrival_date_month","arrival_date_month_n",
                        "reservation_status","reservation_status_date",
                        "arrival_date_day_of_month", "arrival_date_year")
    df = df.with_columns(pl.when(pl.col("reserved_room_type") == pl.col("assigned_room_type")
        ).then(pl.lit(1)).otherwise(pl.lit(0)).alias("same_room_type"))

    # create other country 
    country_counts = df.group_by('country').len().sort('len', descending=True)

    countries = country_counts.filter(pl.col('len') > num_c)['country'].to_list() 
    df = df.with_columns(
        pl.when(pl.col('country').is_in(countries))
        .then(pl.col('country'))
        .otherwise(pl.lit('Other'))
        .alias('country')
    )
        # get categorical columns
    categorical_cols = []
    for col, dtype in zip(df.columns, df.dtypes):
        if dtype == pl.Utf8:
            categorical_cols.append(col)

    # Label encode categorical columns
    label_encoders = {}
    for col in categorical_cols:
        labEnc = LabelEncoder()

        # get unique values and create mapping
        unique_vals = df[col].unique().to_list()
        labEnc.fit(unique_vals)
        label_encoders[col] = labEnc
        
        # mapping
        mapping = {}
        for val in unique_vals:
            mapping[val] = labEnc.transform([val])[0]

        # Apply encoding in Polars
        df = df.with_columns(
            pl.col(col).replace(mapping).alias(col)
        )
    return df, label_encoders

def train_and_metric(df):

    # Convert target and features to numpy for sklearn
    y = df.select("is_canceled").to_numpy().flatten()
    X = df.drop("is_canceled").to_numpy()

    # get feature names
    feature_names = df.drop("is_canceled").columns

    # Train model
    model = RandomForestClassifier(n_estimators= n_estimators, random_state=seed, n_jobs=-1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= test_size, random_state=seed)
    model.fit(X_train, y_train)
    print("Model trained")

    # get predicts
    y_pred = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)

    met = metrics.classification_report(y_test, y_pred, target_names=["Not Canceled", "Canceled"])

    # cross validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)


    scoring = [
        'accuracy',
        'recall',
        'precision',
        'f1',
        'roc_auc'
    ]

    cv_results = cross_validate(
        model, X, y,
        cv=cv,
        scoring=scoring,
        return_train_score=False
    )

    cv_metrics = {
        'accuracy': cv_results['test_accuracy'].mean(),
        'recall': cv_results['test_recall'].mean(),
        'precision': cv_results['test_precision'].mean(),
        'f1': cv_results['test_f1'].mean(),
        'roc_auc': cv_results['test_roc_auc'].mean()
    }


    importance_dict = {}

    for idx in range(len(feature_names)):
        feature_name = feature_names[idx]
        importance_value = model.feature_importances_[idx]
        importance_dict[feature_name] = float(importance_value)


    auc_score = metrics.roc_auc_score(y_test, y_pred_prob[:, 1])
    classification_rep = metrics.classification_report(y_test, y_pred, 
        target_names=["Not Canceled", "Canceled"], output_dict=True)
    
    country_name = df.select("country").unique().to_series().to_list()

    confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_prob[:, 1])
    model_metrics = {
        'auc_score': auc_score,
        'classification_report': classification_rep,
        'confusion_matrix': confusion_matrix,
        'roc_curve': {'fpr': fpr, 'tpr': tpr, 'thresholds': thresholds},
        'feature_importance':  importance_dict,
        'cv_scores': cv_metrics,
        'feature_names': feature_names,
        'country_names': country_name
    }
    return model, model_metrics

def exist(file_path):
    try:
        with open(file_path, 'rb'):
            return True
    except FileNotFoundError:
        return False
    
def main():
    # prepare data
    df, label_encoders = preproc()
    #train model and get metrics
    model, model_metrics = train_and_metric(df)

    # Check if the model already exists
    if not exist("random_forest_model_0.pkl"):
        # Save the model
        joblib.dump(model, "random_forest_model_0.pkl")
    if not exist("model_RF0_metrics.pkl"):
        #save the model metrics
        joblib.dump(model_metrics, "model_RF0_metrics.pkl")
    if not exist("label_encoders_RF0.pkl"):
        # save the label encoders
        joblib.dump(label_encoders, "label_encoders_RF0.pkl")

if __name__ == "__main__":
    main()
