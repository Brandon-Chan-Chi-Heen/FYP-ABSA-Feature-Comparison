import streamlit as st
import nltk
import re
import string
import pandas

# balancing data
from imblearn.over_sampling import SMOTE
from pyloras import LORAS

# data splitting and result reporting
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score, f1_score, classification_report

# feature extraction
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
from joblib import dump, load
import os


data = pandas.read_csv('Tweets-transformed.csv')

# Remove stop words
def remove_stopwords(text):
    try:
        stopword = nltk.corpus.stopwords.words('english')
    except LookupError:
        print('English stopwords not downloaded. Downloading Stopwords...')
        nltk.download('stopwords')
        stopword = nltk.corpus.stopwords.words('english')
    
    text = ' '.join([word for word in text.split() if word not in (stopword)])
    return text

# Remove url  
def remove_url(text):
    url = re.compile(r'https?://\S+|www\.\S+')
    return url.sub(r'',text)

# Remove punct
def remove_punct(text):
    table = str.maketrans('', '', string.punctuation)
    return text.translate(table)

# Remove html 
def remove_html(text):
    html=re.compile(r'<.*?>')
    return html.sub(r'',text)

# Remove @username
def remove_username(text):
    return re.sub('@[^\s]+','',text)

# Remove emojis
def remove_emoji(text):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)


# Decontraction text
def decontraction(text):
    text = re.sub(r"won\'t", " will not", text)
    text = re.sub(r"won\'t've", " will not have", text)
    text = re.sub(r"can\'t", " can not", text)
    text = re.sub(r"don\'t", " do not", text)
    
    text = re.sub(r"can\'t've", " can not have", text)
    text = re.sub(r"ma\'am", " madam", text)
    text = re.sub(r"let\'s", " let us", text)
    text = re.sub(r"ain\'t", " am not", text)
    text = re.sub(r"shan\'t", " shall not", text)
    text = re.sub(r"sha\n't", " shall not", text)
    text = re.sub(r"o\'clock", " of the clock", text)
    text = re.sub(r"y\'all", " you all", text)
    text = re.sub(r"n\'t", " not", text)
    text = re.sub(r"n\'t've", " not have", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'s", " is", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'d've", " would have", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'ll've", " will have", text)
    text = re.sub(r"\'t", " not", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'m", " am", text)
    text = re.sub(r"\'re", " are", text)
    return text  

# Seperate alphanumeric
def seperate_alphanumeric(text):
    words = text
    words = re.findall(r"[^\W\d_]+|\d+", words)
    return " ".join(words)

def cont_rep_char(text):
    tchr = text.group(0) 
    
    if len(tchr) > 1:
        return tchr[0:2] 

def unique_char(rep, text):
    substitute = re.sub(r'(\w)\1+', rep, text)
    return substitute


# replaces the first letter with a space if the letter is not an alphabet
def char(text):
    substitute = re.sub(r'[^a-zA-Z]',' ',text)
    return substitute


# Apply functions on tweets
def preprocess(text):
    preprocessing_functions = [
        remove_username,
        remove_url,
        remove_emoji,
        decontraction,
        seperate_alphanumeric,
        lambda x: unique_char(cont_rep_char,x),
        char,
        lambda x: x.lower(),
        remove_stopwords
    ]
    
    for func in preprocessing_functions:
        text = func(text)
    
    return text

data['final_text'] = data['text'].apply(preprocess)

# st.write(pandas.DataFrame({
#     'text' : data['text'],
#     'final_text' : data['final_text']
#     }
# ))

# split the test data
x_train, x_test, y_train, y_test = train_test_split(data['final_text'], data['airline_sentiment'],train_size = 0.7,test_size = 0.3, random_state=0)

# train feature extractor and model
# @returns tuple[fitted_x_test, featureExtractor, estimator]
def train(featureExtractor, x_train_data, x_test_data, y_train_data, name, balanced=False):
    def train_estimator(x_train_data, y_train_data, name):
        tuning_params = [
            {
                'n_estimators' : [100], 
                'random_state': [0]
            }
        ]

        estimator = None
        if os.path.exists(name + '.joblib'):
            estimator = load(name + '.joblib')
        else:
            adaboost = AdaBoostClassifier()
            grid_search = GridSearchCV(adaboost, tuning_params, scoring='f1_macro', n_jobs=-1)
            grid_search.fit(x_train_data, y_train_data)
            estimator = grid_search.best_estimator_
            dump(grid_search.best_estimator_, name + '.joblib')
        
        return estimator

    if os.path.exists(name + '_extractor.joblib'):
        featureExtractor = load(name + '_extractor.joblib')
    else:
        featureExtractor = featureExtractor.fit(x_train_data)
        dump(featureExtractor, name + '_extractor.joblib')

    if balanced:
        name += '_balanced'
    else:
        name += '_imbalanced'

    fitted_x_train = featureExtractor.transform(x_train_data)
    fitted_x_test = featureExtractor.transform(x_test_data)

    if balanced:
        smote_nc = SMOTE(sampling_strategy="auto", random_state=0)
        # smote_nc = LORAS(random_state=0)
        st.write("Before balancing: ", fitted_x_train.shape, y_train_data.shape)
        fitted_x_train, y_train_data = smote_nc.fit_resample(fitted_x_train, y_train_data)
        st.write("After balancing: ", fitted_x_train.shape, y_train_data.shape)
    
    estimator = train_estimator(fitted_x_train, y_train_data, name)

    return fitted_x_test, featureExtractor, estimator

# get classification report
def getReport(fitted_x, y_test_data, estimator):
    y_pred = estimator.predict(fitted_x)
    return classification_report(y_test_data, y_pred, output_dict=True) 

inputTab, classificationTab = st.tabs(['User Input', "Classification Results"])

with classificationTab:
    
    st.text("Imbalanced Data")
    imbalancedCol1, imbalancedCol2 = st.columns(2)

    with imbalancedCol1:
        # bigram/trigram
        x_test_ngram, ngram_extractor, ngram_estimator = train(CountVectorizer(ngram_range=(2,3)), x_train, x_test, y_train, 'ngram', balanced=False)
        ngram_report = getReport(x_test_ngram, y_test, ngram_estimator)
        st.write("n-gram report")
        st.dataframe(pandas.DataFrame(ngram_report).transpose())

    with imbalancedCol2:
        # Bag of words
        x_test_bagOfWords, bagOfWords_extractor, bagOfWords_estimator = train(CountVectorizer(), x_train, x_test, y_train, 'bagOfWords', balanced=False)
        bagOfWords_report = getReport(x_test_bagOfWords, y_test, bagOfWords_estimator)
        st.write("Bag Of Words")
        st.dataframe(pandas.DataFrame(bagOfWords_report).transpose())

    
    st.text("Balanced Data")
    balancedCol1, balancedCol2 = st.columns(2)

    with balancedCol1:
        # bigram/trigram
        x_test_ngram_balanced, ngram_extractor_balanced, ngram_estimator_balanced = train(CountVectorizer(ngram_range=(2,3)), x_train, x_test, y_train, 'ngram', balanced=True)
        ngram_report_balanced = getReport(x_test_ngram_balanced, y_test, ngram_estimator_balanced)
        st.write("n-gram report")
        st.dataframe(pandas.DataFrame(ngram_report_balanced).transpose())

    with balancedCol2:
        # Bag of words
        x_test_bagOfWords_balanced, bagOfWords_extractor_balanced, bagOfWords_estimator_balanced = train(CountVectorizer(), x_train, x_test, y_train, 'bagOfWords', balanced=True)
        bagOfWords_report_balanced = getReport(x_test_bagOfWords_balanced, y_test, bagOfWords_estimator_balanced)
        st.write("Bag Of Words")
        st.dataframe(pandas.DataFrame(bagOfWords_report_balanced).transpose())

# imbalanced prediction

def getPrediction(text, extractor, estimator):
    if not text:
        return ""
    
    cleanedText = preprocess(text)
    extractedText = extractor.transform([cleanedText])
    prediction = estimator.predict(extractedText)
    return prediction

userInput = inputTab.text_input("Enter your airline review here")

inputTab.text("CountVec Prediction: " + getPrediction(userInput, ngram_extractor, ngram_estimator))
inputTab.text("BagOfWords Prediction: " + getPrediction(userInput, bagOfWords_extractor, bagOfWords_estimator))

# balanced prediction
# inputTab.text("Some other algorithm Prediction: " + getPrediction(userInput, bagOfWords_extractor, bagOfWords_estimator))



