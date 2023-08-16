import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load and preprocess the dataset
df = pd.read_csv("/content/drive/MyDrive/Health.csv")
df['Symptoms'] = df['Symptoms'].str.lower()  # Convert to lowercase

# Define disease labels and map them to integers
disease_mapping = {
    'HeartDisease': 0,
    'JointDislocation': 1,
    'LowBp': 2,
    'SnakeBite': 3
}

# Preprocess text data
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    words = word_tokenize(text)
    words = [lemmatizer.lemmatize(word) for word in words if word.isalpha()]
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

df['preprocessed_Symptoms'] = df['Symptoms'].apply(preprocess_text)

# Create TF-IDF features
vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(df['preprocessed_Symptoms'])
y = df['Disease'].map(disease_mapping)

# Train a Naive Bayes classifier
clf = MultinomialNB()
clf.fit(X, y)

# User input and prediction
while True:
    user_input = input("Enter your Symptoms (or 'exit' to quit): ").lower()
    if user_input == 'exit':
        break
    user_input = preprocess_text(user_input)
    user_input_tfidf = vectorizer.transform([user_input])
    predicted_disease = clf.predict(user_input_tfidf)[0]
    for disease, label in disease_mapping.items():
        if label == predicted_disease:
            print(f"Predicted disease: {disease}")
            break

# This is text which we will be giving
#The person is having the shortness of the breath, and  he is having the irregular heartbeat, and he is sweting too much, his facial expression is also changed 

