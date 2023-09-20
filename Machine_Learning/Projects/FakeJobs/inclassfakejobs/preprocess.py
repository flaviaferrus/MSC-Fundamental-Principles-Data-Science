import numpy as np
import pandas as pd
import re
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
import nltk
from imblearn.combine import SMOTETomek
import string
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

nltk.download('stopwords')

stop_words = set(stopwords.words("english"))
default_stemmer = PorterStemmer()
default_stopwords = stopwords.words('english')
default_tokenizer = RegexpTokenizer(r"\w+")

# # Making a dictionary of the words and their vector representation

embeddings_index = {}
f = open('glove/glove.840B.300d.txt')
for line in f:
    values = line.split(' ')
    word = values[0]  # # The first entry is the word
    coefs = np.asarray(values[1:],
                       dtype='float32')  # # These are the vectors representing the embedding for the word
    embeddings_index[word] = coefs
f.close()

glove_words = set(embeddings_index.keys())

# https://gist.github.com/sebleier/554280
# we are removing the words from the stop words list
stopwords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves',
             'you', "you're", "you've",
             "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves',
             'he', 'him', 'his', 'himself',
             'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its',
             'itself', 'they', 'them', 'their',
             'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this',
             'that', "that'll", 'these', 'those',
             'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have',
             'has', 'had', 'having', 'do', 'does',
             'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or',
             'because', 'as', 'until', 'while', 'of',
             'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into',
             'through', 'during', 'before', 'after',
             'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on',
             'off', 'over', 'under', 'again', 'further',
             'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how',
             'all', 'any', 'both', 'each', 'few', 'more',
             'most', 'other', 'some', 'such', 'only', 'own', 'same', 'so',
             'than', 'too', 'very',
             's', 't', 'can', 'will', 'just', 'don', "don't", 'should',
             "should've", 'now', 'd', 'll', 'm', 'o', 're',
             've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn',
             "didn't", 'doesn', "doesn't", 'hadn',
             "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't",
             'ma', 'mightn', "mightn't", 'mustn',
             "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn',
             "shouldn't", 'wasn', "wasn't", 'weren', "weren't",
             'won', "won't", 'wouldn', "wouldn't"]


# # Defining the utility functions

def remove_URL(text):
    url = re.compile(r'https?://\S+|www\.\S+')
    return url.sub(r'', str(text))


def remove_emoji(text):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', str(text))


def remove_html(text):
    html = re.compile(r'<.*?>')
    return html.sub(r'', str(text))


def remove_punctuation(text):
    table = str.maketrans('', '', string.punctuation)
    return text.translate(table)


def decontracted(phrase):
    # specific
    phrase = re.sub(r"won't", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)

    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase


def final_preprocess(text):
    text = text.replace('\\r', ' ')
    text = text.replace('\\"', ' ')
    text = text.replace('\\n', ' ')
    text = re.sub('[^A-Za-z0-9]+', ' ', text)
    text = ' '.join(e for e in text.split() if e.lower() not in stopwords)
    text = text.lower()
    ps = PorterStemmer()
    text = ps.stem(text)
    return text


def convert_sen_to_vec(sentence):
    vector = np.zeros(300)  # as word vectors are of zero length
    cnt_words = 0  # num of words with a valid vector in the sentence
    for _word in sentence.split():
        if _word in glove_words:
            vector += embeddings_index[_word]
            cnt_words += 1
    if cnt_words != 0:
        vector /= cnt_words
    return vector


def prepare_train(data: pd.DataFrame):
    data = data.replace(np.nan, '', regex=True)
    data['text'] = data[
        ['title', 'department', 'company_profile', 'description',
         'requirements', 'benefits']].apply(lambda x: ' '.join(x), axis=1)
    data.drop(['job_id', 'location', 'title', 'salary_range', 'department',
               'salary_range', 'company_profile', 'description',
               'requirements', 'benefits'], axis=1, inplace=True)
    try:
        data.drop(['required_doughnuts_comsumption'], axis=1, inplace=True)
    except ValueError:
        data.drop(['doughnuts_comsumption'], axis=1, inplace=True)

    label_columns = ['telecommuting', 'has_company_logo', 'has_questions',
                     'employment_type', 'required_experience',
                     'required_education', 'industry', 'function']
    lb_make = LabelEncoder()
    for i in label_columns:
        data[i] = lb_make.fit_transform(data[i])

    data['text'] = remove_URL(str(data['text']))
    data['text'] = remove_emoji(str(data['text']))
    data['text'] = remove_html(str(data['text']))
    data['text'] = remove_punctuation(str(data['text']))
    data['text'] = final_preprocess(str(data['text']))

    converted_data = []

    for i in range(0, data.shape[0]):
        converted_data.append(convert_sen_to_vec(data['text'][i]))

    _1 = pd.DataFrame(converted_data)

    scaler = StandardScaler()

    data[['required_education', 'required_experience',
          'employment_type']] = scaler.fit_transform(
        data[['required_education', 'required_experience', 'employment_type']])
    data.drop(["text"], axis=1, inplace=True)
    main_data = pd.concat([_1, data], axis=1)
    print(main_data.head())
    X = main_data[[c_ for c_ in data.columns if
                   str(c_) != 'fraudulent']]
    Y = main_data[['fraudulent']]

    smk = SMOTETomek(random_state=42)
    X_res, Y_res = smk.fit_sample(X, Y)

    return X_res, Y_res, lb_make, scaler


def prepare_test(data: pd.DataFrame, lb_make, scaler):
    data = data.replace(np.nan, '', regex=True)
    data['text'] = data[
        ['title', 'department', 'company_profile', 'description',
         'requirements', 'benefits']].apply(lambda x: ' '.join(x), axis=1)
    data.drop(['job_id', 'location', 'title', 'salary_range', 'department',
               'salary_range', 'company_profile', 'description',
               'requirements', 'benefits'], axis=1, inplace=True)
    try:
        data.drop(['required_doughnuts_comsumption'], axis=1, inplace=True)
    except ValueError:
        data.drop(['doughnuts_comsumption'], axis=1, inplace=True)

    label_columns = ['telecommuting', 'has_company_logo', 'has_questions',
                     'employment_type', 'required_experience',
                     'required_education', 'industry', 'function']

    for i in label_columns:
        data[i] = lb_make.transform(data[i])

    data['text'] = remove_URL(str(data['text']))
    data['text'] = remove_emoji(str(data['text']))
    data['text'] = remove_html(str(data['text']))
    data['text'] = remove_punctuation(str(data['text']))
    data['text'] = final_preprocess(str(data['text']))

    converted_data = []

    for i in range(0, data.shape[0]):
        converted_data.append(convert_sen_to_vec(data['text'][i]))

    _1 = pd.DataFrame(converted_data)

    data[['required_education', 'required_experience',
          'employment_type']] = scaler.transform(
        data[['required_education', 'required_experience', 'employment_type']])
    data.drop(["text"], axis=1, inplace=True)
    main_data = pd.concat([_1, data], axis=1)

    X = main_data[[c_ for c_ in data.columns if
                   str(c_) != 'fraudulent']]
    # Y = main_data[['fraudulent']]

    # smk = SMOTETomek(random_state=42)
    # X_res, Y_res = smk.fit_sample(X, Y)

    return X


def main() -> None:
    train = pd.read_csv("input/train.csv", header=0, index_col=0)
    test = pd.read_csv("input/test.csv", header=0, index_col=0)
    features, labels, label_encoder, scaler = prepare_train(train)
    test_processed = prepare_test(test, label_encoder, scaler)
    pd.DataFrame(features).to_csv('x_train.csv')
    pd.DataFrame(labels).to_csv('y_train.csv')
    pd.DataFrame(test_processed).to_csv('x_test.csv')
