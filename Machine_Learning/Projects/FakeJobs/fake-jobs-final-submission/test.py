import numpy as np
import pandas as pd
from advanced import Logger
from auxiliary import save_data
from nlp import nlp_processing, predict, load_data, classifiers
from typing import List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


long_features = ['title', 'company_profile', 'industry', ]  # 'function', 'description']  # 'description does not add value, but it lowers; same for 'function', in lower measure
onehot_features = ['required_experience', 'employment_type']  # 'required_education'
other_features = ['country', 'state', 'city', 'department', 'requirements']  # 'benefits'
# numerical_features = ['salary_range', 'required_doughnuts_comsumption']
# total_features = long_features + onehot_features + other_features
total_features = ['title', 'company_profile', 'industry', 'country', 'department']
logger = Logger()


############################################################################
# MAIN
############################################################################

# TODO: add the one_hot features, one_hot encoded instead as pure text!

def add_features():
    """
    Rank the best training in function of the numbers of features added.
    Every feature added is parsed as text; then tokenized & word embedded.
    """
    added_features = []
    performance_df: pd.DataFrame = pd.DataFrame()
    logger.info("Starting the feature selection process")
    for feature in total_features:
        added_features.append(feature)

        logger.info(
            f"Training the model using {len(added_features)} features")
        new_df: pd.DataFrame = train_rf_model(added_features)

        # appending it and continuing
        performance_df = pd.concat(
            [performance_df, new_df], axis=0)
        print(performance_df)
        save_data(performance_df, 'feature_importance')


###########################################################################
# AUXILIARY
###########################################################################

def train_rf_model(columns: List[str]) -> pd.DataFrame:
    features, labels = load_data()
    features = nlp_processing(features, columns)
    # _features = pd.concat(_features, _features['nan_per_sample'], axis=1)  # how does the pipeline would handle it?
    report_df: pd.DataFrame = predict(
        features, labels, print_performance=False)
    report_df.index = np.atleast_1d(len(columns))
    return report_df


#########################################################################
# TRAINING
#########################################################################

def make_prediction(columns: List[str],
                    model: str = 'RandomForestClassifier') -> \
        Tuple[pd.DataFrame, pd.DataFrame]:

    train_features, train_labels = load_data()
    test_features, _ = load_data(test=True)
    test_labels = pd.Series(np.zeros(len(test_features)))

    x: pd.DataFrame = nlp_processing(
        pd.concat([train_features, test_features], axis=0), columns)
    y: pd.Series = pd.concat([train_labels['fraudulent'], test_labels], axis=0)

    logger.info("Preparing the model for inference")
    logger.debug("\t\tCreating a pipeline: the data is first vectorized "
                 "using a Word Embedding")
    bundled_pipeline = Pipeline(
        [("tfidf", TfidfVectorizer()), ("clf", classifiers[model])])

    logger.debug("\t\t Splitting the data into train/test")

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, train_size=len(train_features), random_state=True, shuffle=False)

    logger.debug("\t\t Training the model")
    bundled_pipeline.fit(x_train.T, y_train.ravel())
    logger.debug("\t\t Using the model to predict")
    y_hat = bundled_pipeline.predict(x_test.T)
    save_data(y_hat, 'y_hat')

    return y_hat


if __name__ == '__main__':
    # add_features()
    _columns = total_features
    make_prediction(_columns)

    # model selection # for the desired columns...
    # from nlp import find_best_model
    # find_best_model(_columns)
