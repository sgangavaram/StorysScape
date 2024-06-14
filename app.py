import pickle
import re
import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
from flask import Flask, request, render_template

nltk.download('stopwords')
nltk.download('wordnet')

def cleantext(text):
    text = re.sub("'\''", "", text)
    text = re.sub("[^a-zA-Z]", " ", text)
    text = ' '.join(text.split())
    text = text.lower()
    return text

def removestopwords(text):
    stop_words = set(stopwords.words('english'))
    removedstopword = [word for word in text.split() if word not in stop_words]
    return ' '.join(removedstopword)

def lemmatizing(text):
    lemma = WordNetLemmatizer()
    stemSentence = ""
    for word in text.split():
        stem = lemma.lemmatize(word)
        stemSentence += stem
        stemSentence += " "
    stemSentence = stemSentence.strip()
    return stemSentence

def stemming(text):
    stemmer = PorterStemmer()
    stemmed_sentence = ""
    for word in text.split():
        stem = stemmer.stem(word)
        stemmed_sentence += stem
        stemmed_sentence += " "
    stemmed_sentence = stemmed_sentence.strip()
    return stemmed_sentence

def test(text, model, tfidf_vectorizer):
    text = cleantext(text)
    text = removestopwords(text)
    text = lemmatizing(text)
    text = stemming(text)
    text_vector = tfidf_vectorizer.transform([text])
    predicted = model.predict(text_vector)
    newmapper = {0: 'Fantasy', 1: 'Science Fiction', 2: 'Crime Fiction',
                 3: 'Historical novel', 4: 'Horror', 5: 'Thriller'}
    return newmapper[predicted[0]]

with open('bookgenremodel.pkl', 'rb') as file:
    model = pickle.load(file)

with open('tfdifvector.pkl', 'rb') as file:
    tfidf_vectorizer = pickle.load(file)

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def hello_world():
    if request.method == 'POST':
        text = request.form['summary']
        prediction = test(text, model, tfidf_vectorizer)
        return render_template('index.html', genre=prediction, text=str(text)[:100], showresult=True)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8082)
