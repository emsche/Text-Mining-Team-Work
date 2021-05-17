from nltk.stem.snowball import SnowballStemmer
from torchtext.data import get_tokenizer
from sklearn.feature_extraction.text import CountVectorizer
stemmer = SnowballStemmer("english")
tokenizer = get_tokenizer("basic_english")
cv = CountVectorizer()

def stem_tkn(text):
    return [stemmer.stem(w) for w in tokenizer(tweet)]

def bow(input_text, corpus_text):
    cv.fit_transform(corpus_text)
    X = cv.transform(input_text).toarray()
    return X

#corpus = ['This is the first document.',
          #'This is the second second document.', 
          #'And the third one.', 'Is this the first document?']

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

baseline_vectorizer = TfidfVectorizer(tokenizer=stem_tkn)
baseline_model = LogisticRegression(max_iter=300, n_jobs=-1)
baseline_pipeline = Pipeline(steps=[
    ('vectorizer', baseline_vectorizer),
    ('model', baseline_model)
])