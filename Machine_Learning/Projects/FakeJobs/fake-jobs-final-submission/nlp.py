import os.path

import pandas as pd
import time

import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from warnings import catch_warnings, simplefilter
from typing import List, Tuple
from auxiliary import reduce_mem_usage
from preprocessing import location_extraction, empty_strings_as_nan, \
    add_nan_per_sample
from advanced import Logger
bow_columns: List[str] = ['title', 'function', 'company_profile']

logger = Logger()


def _set_classifiers() -> dict:
    """
    In a preliminary analysis, the best found models were:

    5   RandomForestClassifier     0.65  0.968044     0.008684
    4           LGBMClassifier      0.3  0.966948     0.004576
    3            XGBClassifier      0.4  0.966557     0.003042
    1                LinearSVC     0.05  0.959178     0.008850
    10         RidgeClassifier     0.06  0.958440     0.012945
    13             BernoulliNB     0.04  0.951981     0.007133
    8       AdaBoostClassifier     0.31  0.945582     0.014136
    2            MultinomialNB     0.04  0.938093     0.012422
    11           SGDClassifier     0.05  0.936538     0.011502
    9     KNeighborsClassifier     0.15  0.923557     0.003414
    12       BaggingClassifier     2.55  0.895672     0.017097
    7      ExtraTreeClassifier     0.05  0.852488     0.014914
    6   DecisionTreeClassifier     0.55  0.849216     0.017311
    0          DummyClassifier     0.06  0.500000     0.000000


    """

    from sklearn.svm import LinearSVC
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    # from sklearn.dummy import DummyClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.tree import ExtraTreeClassifier
    from sklearn.linear_model import RidgeClassifier
    from sklearn.linear_model import SGDClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.ensemble import BaggingClassifier
    from sklearn.naive_bayes import BernoulliNB
    from sklearn.naive_bayes import MultinomialNB

    _classifiers = {}
    _classifiers.update({"XGBCustom": XGBClassifier(
        learning_rate=0.1,
        min_child_weight=1,
        max_depth=6,
        n_estimators=1300,
        gamma=0,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='binary:logistic',
        nthread=-1,
        reg_alpha=0.05,
        eval_metric='auc',
        scale_pos_weight=1)})

    # _classifiers.update({"DummyClassifier": DummyClassifier(strategy='most_frequent')})

    _classifiers.update({"RandomForestClassifier": RandomForestClassifier()})
    _classifiers.update({"LGBMClassifier": LGBMClassifier()})
    _classifiers.update({"XGBClassifier": XGBClassifier(eval_metric='auc')})
    _classifiers.update({"LinearSVC": LinearSVC()})
    _classifiers.update({"RidgeClassifier": RidgeClassifier()})
    _classifiers.update({"BernoulliNB": BernoulliNB()})
    _classifiers.update({"AdaBoostClassifier": AdaBoostClassifier()})
    _classifiers.update({"MultinomialNB": MultinomialNB()})
    _classifiers.update({"SGDClassifier": SGDClassifier()})
    _classifiers.update({"KNeighborsClassifier": KNeighborsClassifier()})
    _classifiers.update({"BaggingClassifier": BaggingClassifier()})
    _classifiers.update({"ExtraTreeClassifier": ExtraTreeClassifier()})
    _classifiers.update({"DecisionTreeClassifier": DecisionTreeClassifier()})

    return _classifiers


classifiers = _set_classifiers()


def predict(features, labels, model: str = 'RandomForestClassifier',
            print_performance: bool = True) -> pd.DataFrame:
    y_test, y_hat = fast_training(
        features['tokenized'], labels['fraudulent'], model)
    report_df = report_metrics(y_test, y_hat, print_performance=print_performance)
    return report_df


def find_best_model(columns: List[str] = None) -> None:
    if columns is None:
        columns = bow_columns
    logger.info("Finding best model for this set up")
    features, labels = load_data()
    features = nlp_processing(features, columns)
    # we train the model just with the bow columns to see how it performs...
    model_list: pd.DataFrame = model_selection(
        features['tokenized'], labels['fraudulent'])
    print(model_list)


def model_selection(x: pd.Series, y: pd.Series) -> pd.DataFrame:
    df_models = pd.DataFrame(
        columns=['model', 'run_time', 'roc_auc', 'roc_auc_std'])
    logger.debug("\tAssessing and finding the best models' performance")
    for key in classifiers:
        start_time = time.time()
        logger.debug(f"\t Training classifier {key}")

        pipeline = Pipeline(
            [("tfidf", TfidfVectorizer()), ("clf", classifiers[key])])
        with catch_warnings():
            simplefilter('ignore')
            cv = cross_val_score(pipeline, x, y, cv=5, scoring='roc_auc')
            row = {'model': key,
                   'run_time': format(round((time.time() - start_time) / 60, 2)),
                   'roc_auc': cv.mean(),
                   'roc_auc_std': cv.std(),
                   }
            # df_models = pd.concat([df_models, row], axis=0)
            df_models = df_models.append(row, ignore_index=True)

    df_models = df_models.sort_values(by='roc_auc', ascending=False)
    return df_models


def fast_training(x: pd.Series, y: pd.Series,
                  model: str = 'RandomForestClassifier') \
        -> Tuple[pd.DataFrame, pd.DataFrame]:
    logger.debug("\tStarting the training")
    logger.debug("\t\t Splitting the data into train/test")
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.3, random_state=1, shuffle=True)
    logger.debug("\t\t Creating a pipeline: the data is first vectorized "
                 "using a Word Embedding")
    bundled_pipeline = Pipeline(
        [("tfidf", TfidfVectorizer()), ("clf", classifiers[model])])
    logger.debug("\t\t Training the model")
    bundled_pipeline.fit(x_train, y_train)
    logger.debug("\t\t Using the model to predict")
    y_hat = bundled_pipeline.predict(x_test)

    return y_test, y_hat


def report_metrics(y_obs, y_hat, print_performance: bool = False) -> pd.DataFrame:
    logger.debug("\tReporting model's performance")
    report_dict = classification_report(y_obs, y_hat)
    report_df = _classification_report_as_dataframe(report_dict)

    if print_performance:
        print(report_dict)
        print('Accuracy:', accuracy_score(y_obs, y_hat))
        print('F1 score:', f1_score(y_obs, y_hat))
        print('ROC/AUC score:', roc_auc_score(y_obs, y_hat))
    # we just keep the validation class punctuation
    report_df = report_df.iloc[1, :].to_frame().transpose().sort_values(
        by=['f1_score'], ascending=False)
    return report_df


def _classification_report_as_dataframe(report: str) -> pd.DataFrame:
    report_df = pd.DataFrame()
    lines = report.split('\n')
    for i, line in enumerate(lines[2:4]):
        row = {}
        row_data = line.split('      ')
        row['class'] = i  # row_data[0]
        row['precision'] = [float(row_data[1])]
        row['recall'] = [float(row_data[2])]
        row['f1_score'] = [float(row_data[3])]
        row['support'] = [float(row_data[4])]
        with catch_warnings():
            simplefilter('ignore')
            report_df = report_df.append(pd.DataFrame.from_dict(row))
    return report_df


############################################################################
# DATA PROCESSING
############################################################################

def nlp_processing(features: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    logger.debug("\tPerforming the NLP processing")
    features['all_text'] = merge_into_string(features[columns])
    features['all_text'] = punctuation_to_features(features['all_text'])
    features['tokenized'] = tokenize(features['all_text'])
    features['tokenized'] = remove_stopwords(features['tokenized'])
    features['tokenized'] = apply_stemming(features['tokenized'])
    features['tokenized'] = rejoin_words(features['tokenized'])

    return features


def load_data(test: bool = False):
    folder: str = 'input'
    name: str = 'test.csv' if test else 'train.csv'

    logger.debug(f"\tLoading the input {'test' if test else 'train'} dataset.")
    data: pd.DataFrame = pd.read_csv(
        f'{folder}/{name}', sep=',', header=0, index_col=0)
    data = reduce_mem_usage(data)

    if test:
        data = data.rename(columns={
            'doughnuts_comsumption': 'required_doughnuts_comsumption'})

    features = data[[c_ for c_ in data.columns if c_ != 'fraudulent']]

    # NECESSARY PREPROCESSING (nan removal & encoding into strings)
    features = location_extraction(features)
    features = empty_strings_as_nan(features)
    features['nan_per_sample'] = add_nan_per_sample(features)

    # No further preprocess, text tokenizer can handle it

    try:
        labels = data[['fraudulent']]
    except KeyError:
        labels = None  # test = True

    return features, labels


def merge_into_string(features: pd.DataFrame) -> pd.Series:
    logger.debug(
        "\t Merging into the categorical fields into one single string "
        "(and treating nan as empty strings).")

    def _merge(text_df):
        return " ".join(text_df.where(~pd.isna(text_df), other='').astype(str))

    return features.apply(
        lambda x: _merge(x), axis=1)  # sum([features[c_] for c_ in bow_columns])


def tokenize(df: pd.Series) -> pd.Series:
    """
        Tokenizes a Pandas series returning a list of tokens for each sample
        (in form of a new pd.Series).

        Parameters
        ----------
        df: pd.Series
            Pandas dataframe column containing the text to tokenize

        Returns
        -------
        tokens (pd.Series)
            pd.Series consisting in a tokenized list of strings
            containing the text's words
        """
    logger.debug("\t\t Tokenizing the string feature into list of words")
    if not os.path.isdir('/home/gcastro/nltk_data/tokenizers/punkt'):
        logger.debug("\t\t\t Downloading the vocabulary to tokenize")
        nltk.download('punkt')  # it must be firstly run!

    def _tokenize(column):
        tokens = nltk.word_tokenize(column)
        return [w for w in tokens if w.isalpha()]

    return df.apply(lambda x: _tokenize(x))


def punctuation_to_features(df: pd.Series) -> pd.Series:
    """
    Identify punctuation within a column and convert to a text representation.
    Parameters
    ----------
    df: pd.Series
        Pandas Series containing the text.

    Returns
    -------
    df: pd.Series
        Original column with punctuation converted to text.
        E.g. "Wow! > "Wow exclamation"

    """
    logger.debug("\t\t Adding punctuation marks a categorical features")
    df = df.replace('!', ' exclamation ')
    df = df.replace('?', ' question ')
    df = df.replace('\'', ' quotation ')
    df = df.replace('\"', ' quotation ')
    return df


def remove_stopwords(token_df: pd.Series) -> pd.Series:
    """
    Return a list of tokens with English stopwords removed.

    Parameters
    ----------
    token_df: pd.Series
        Pandas Series of tokenized data

    Returns
    -------
    tokens (pd.Series)
        Tokenized list with stopwords removed.

    """
    logger.debug("\t\t Removing the english stopwords")
    if not os.path.isdir('/home/gcastro/nltk_data/corpora/stopwords'):
        logger.debug("\t\t\t Downloading the stopwords vocabulary")
        nltk.download('stopwords')

    def _remove(_text: List[str]) -> List[str]:
        stops = set(stopwords.words("english"))
        return [word for word in _text if word not in stops]

    return token_df.apply(lambda x: _remove(x))


def apply_stemming(token_df: pd.Series) -> pd.Series:
    """
    Return a pd.Series containing the list of tokens with Porter
    stemming applied.

    Parameters
    ----------
    token_df: pd.Series
        Pandas dataframe column of tokenized data
        (with stopwords already removed)

    Returns
    -------
        tokens (pd.Series): Tokenized list with words Porter stemmed.
    """
    logger.debug("\t\t Applying stemming: reducing the words to its roots")

    def _stemming(_text: List[str]) -> List[str]:
        stemmer = PorterStemmer()
        return [stemmer.stem(word).lower() for word in _text]

    return token_df.apply(lambda x: _stemming(x))


def rejoin_words(token_df: pd.Series) -> pd.Series:
    logger.debug("\t\t Rejoining the tokens list into a long string feature")
    return token_df.apply(lambda x: " ".join(x))


if __name__ == '__main__':
    _features, _labels = load_data()
    _features = nlp_processing(_features, columns=bow_columns)
    # find_best_model()
    predict(_features, _labels, 'RandomForestClassifier', print_performance=True)
    # predict(_features, _labels, 'XGBCustom', print_performance=True)
