import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import spacy
import json
# Sample data
with open('train_data.json') as station_list:
    station_data = json.load(station_list)

def convert_json(json_data):
    data = []
    for item in json_data:
        new_item = (item[0], item[1])
        data.append(new_item)
    return data

# Convert JSON data to desired format
data = convert_json(station_data)

X_train = [text for text, label in data]
y_train = [label for text, label in data]

# Convert text data into numerical features using bag-of-words encoding
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)

# Train a simple logistic regression model
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# Function to extract departure and destination from text
def extract_departure_destination(text):
    # Use spaCy for NER
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    
    departure = None
    destination = None

    # Custom NER rules or patterns
    for i, token in enumerate(doc):
        if token.text.lower() == "from" and token.dep_ == "prep":
            departure = doc[i + 1].text
        elif token.text.lower() == "to" and token.dep_ == "prep":
            destination = doc[i + 1].text

    # If departure or destination is still None, fallback to named entities
    if not departure or not destination:
        for ent in doc.ents:
            if (ent.label_ == "GPE" or ent.label_ == "ORG") and not departure:
                departure = ent.text
            elif (ent.label_ == "GPE" or ent.label_ == "ORG") and not destination:
                destination = ent.text

    return departure, destination
