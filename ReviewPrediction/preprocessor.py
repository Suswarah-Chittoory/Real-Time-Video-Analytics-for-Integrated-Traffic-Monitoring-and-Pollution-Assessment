import nltk
nltk.download('stopwords')
# Downloading wordnet before applying Lemmatizer
nltk.download('wordnet')
nltk.download('omw-1.4')

import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Define text preprocessing function
def preprocessor(text):
    # Removing special characters and digits
    letters_only = re.sub("[^a-zA-Z]", " ", text)
    # change sentence to lower case
    letters_only = letters_only.lower()
    # tokenize into words
    words = letters_only.split()
    # remove stop words
    words = [word for word in words if word not in stopwords.words("english")]
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    return " ".join(words)
