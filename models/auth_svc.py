#import libraries

import pandas as pd
from sklearn.model_selection import train_test_split

print("preprocessing...")

df_raw = pd.read_csv("../datasets/Philippine Fake News Corpus.csv")

df_real = df_raw[df_raw['Label'] =='Credible']
df_real.reset_index(inplace=True, drop=True)
df_real = df_real[0:7656] #undersampling to get equal sample size for credible news and not credible news (fake news)

df_fake = df_raw[df_raw['Label'] =='Not Credible']
df_fake.reset_index(inplace=True, drop=True)
del df_raw

df_undersampled = pd.concat([df_real, df_fake])
df_undersampled.reset_index(inplace=True, drop=True)
del df_real
del df_fake

df_undersampled.drop(['Authors', 'Date', 'URL', 'Brand'], inplace = True, axis = 1)

df_undersampled['binary_label'] = df_undersampled['Label'].apply(lambda c: 0 if c == 'Credible' else 1)
df_undersampled = df_undersampled.sample(frac=1, random_state=1) #shuffle the dataset
df_undersampled.reset_index(inplace=True, drop=True)

df_undersampled["whole_article"] = df_undersampled["Headline"] + df_undersampled["Content"]

df_undersampled['cleaned_article'] = df_undersampled['whole_article']#.apply(complete_preprocessing)

y = df_undersampled["binary_label"]
x = df_undersampled["cleaned_article"]
x_train,x_test,y_train,y_test = train_test_split(x,y, test_size = 0.3, # 30% of data reserved for testing
                                                 random_state=1, shuffle=True, stratify = y)
del df_undersampled
print("preprocessing complete.")

print("extracting features...")
#feature extraction using tf-idf
from sklearn.feature_extraction.text import TfidfVectorizer
tf_vectorizer = TfidfVectorizer()

#x_train uses fit_transform(); fit will only be done on training data
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

from sklearn.metrics import classification_report, accuracy_score

# predicting testing data split
test_predictions = model.predict(x_test_vector)

# getting accuracy
print("getting accuracy...")
print("Accuracy: ", accuracy_score(y_test, test_predictions, normalize=True))

print("making classification report...")
print(classification_report(y_test, test_predictions, target_names=['Not Fake News','Fake News']))
print("model evaluation complete.")

print("exporting model...")
import pickle
import os

if not(os.path.exists("auth_svc.pkl")):
    pickle.dump(model, open("auth_svc.pkl", "wb"))
    print("Dumped model to auth_svc.pkl")
if not (os.path.exists("../preprocessing/auth_svc_vectorizer.pkl")):
    pickle.dump(tf_vectorizer, open("../preprocessing/auth_svc_vectorizer.pkl", "wb"))
    print("Dumped vectorizer to auth_svc_vectorizer.pkl")

model_size = os.path.getsize('auth_svc.pkl')
transformer_size = os.path.getsize('auth_svc_vectorizer.pkl')

print(str(int(model_size/1000000)) + " mb")
print(str(int(transformer_size/1000000)) + " mb")

