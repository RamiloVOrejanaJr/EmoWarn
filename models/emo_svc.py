#import libraries

import pandas as pd
from sklearn.model_selection import train_test_split

print("preprocessing...")

#Importing Dataset

#Order of emotions: Joy, Sadness, Anger, Fear, Surprise

emotions = ['joy', 'sadness', 'anger', 'fear', 'surprise']

df = pd.read_csv('../datasets/cleaned_ren10k.csv')

#df['cleaned_article'] = df['whole_article'].apply(complete_preprocessing)

"""#Feature Extraction"""

#Splitting Dataset
x = df['whole_article']
y = df['emotion_int']

from sklearn.feature_extraction.text import TfidfVectorizer

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state=1, stratify =y)  # 30% of data reserved for testing

tf_vectorizer = TfidfVectorizer()

#feature extraction using tf-idf

#x_train uses fit_transform(); fit will only be done on training data
print(x_train[1])
print(x_train[1].shape)
print(type(x_train[1]))
x_train_vector = tf_vectorizer.fit_transform(x_train)

#x_test; only transform () will be done on testing data
x_test_vector = tf_vectorizer.transform(x_test)
print("feature extraction complete.")

print("model training...")
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

#making linearsvc model
model = LinearSVC()
model = CalibratedClassifierCV(model)

#training linearsvc model on training data
model.fit(x_train_vector, y_train)

print("model training complete.")

print("evaluating model...")

from sklearn.metrics import classification_report, accuracy_score #,confusion_matrix

# predicting testing data split
test_predictions = model.predict(x_test_vector)

# getting accuracy
print("getting accuracy...")
print("Accuracy: ", accuracy_score(y_test, test_predictions, normalize=True))

print("making classification report...")
print(classification_report(y_test, test_predictions, target_names=["Joy", "Sadness", "Anger", "Fear", "Surprise"]))
print("model evaluation complete.")

print(x_test_vector[1])
print(x_test_vector[1].shape)
print(type(x_test_vector[1]))

y_proba = model.predict_proba(x_test_vector[1])
result = model.predict(x_test_vector[1])

#normalizing to integers
print("normalizing probabilities...")
def normalize_to_int(probability_distribution):
    probability_distribution = [round(probability * 100) for probability in probability_distribution[0]]
    least_value = probability_distribution.index(min(probability_distribution))
    rem = 100 - sum(probability_distribution)
    probability_distribution[least_value] += rem

    return probability_distribution

y_proba = normalize_to_int(y_proba)

print(y_proba)
print(sum(y_proba))
print(result)

print("normalization complete.")


print("exporting model...")
import pickle
import os

if not(os.path.exists("emo_svc.pkl")):
    pickle.dump(model, open("emo_svc.pkl", "wb"))
    print("Dumped model to emo_svc.pkl")
if not (os.path.exists("../preprocessing/emo_svc_vectorizer.pkl")):
    pickle.dump(tf_vectorizer, open("../preprocessing/emo_svc_vectorizer.pkl", "wb"))
    print("Dumped vectorizer to emo_svc_vectorizer.pkl")

model_size = os.path.getsize("emo_svc.pkl")
transformer_size = os.path.getsize("emo_svc_vectorizer.pkl")

print(str(int(model_size/1000000)) + " mb")
print(str(int(transformer_size/1000000)) + " mb")
print("exporting complete.")


