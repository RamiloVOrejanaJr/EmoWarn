from flask import Flask, request, render_template
from nltk.corpus import stopwords
import tensorflow as tf
import pickle as pkl

#import tf_keras

import requests
import re
print("with checkpoints")

print("finished importing")

pattern = re.compile(r'\b(' + r'|'.join(stopwords.words('english')) + r')\b\s*')

print("finished compiling re pattern")

# define preprocessing method
def complete_preprocessing(article):
    article = str(article).lower()
    article = re.sub('[^a-zA-Z]', ' ', article)
    article = re.sub('\s+[^a-zA-Z]\s+', ' ', article)
    article = pattern.sub('', article)
    return article


# get emotion clasification model
emo_model = tf.keras.models.load_model('emo_model.keras')
file = open('emo_tf_vectorizer.pkl', 'rb')
emo_vectorizer = pkl.load(file)
file.close()

#get fake news classification model
fake_news_model = tf.keras.models.load_model('fake_news_model.keras')
file = open('auth_tf_vectorizer.pkl', 'rb')
fake_news_vectorizer = pkl.load(file)
file.close()


print("finished loading ai models")

print()
# Create flask app
print("test")
flask_app = Flask(__name__)
print("test2")


@flask_app.route("/")
def home():

    data = {'fake': 99,
            'real': 1,
            'joy': 96,
            'sadness': 1,
            'anger': 1,
            'fear': 1,
            'surprise': 1
            }

    '''try:
        # Replace <your_PC_IP> with your actual PC's IP address and port
        pc_server_url = "http://emowarn.ddns.net:5000/data"
        response = requests.get(pc_server_url)
        print("Response type: ", type(response))
        print("Response: ", response)

        # Process the response from the PC server
        if response.status_code == 200:
            data = response.json()
            #return jsonify({"message": "Successfully fetched data", "data": data})
        #else:
            #return jsonify({"error": "Failed to connect to PC server"}), 500
    except Exception as e:
        print(jsonify({"error": str(e)}), 500)'''

    headline_placeholder = "Enter news headline..."
    article_placeholder = "Enter news article..."
    return render_template("index.html", chart_data=data,
                           headline_placeholder=headline_placeholder,
                           article_placeholder=article_placeholder)


@flask_app.route('/instructions')
def instructions():
    return render_template('instructions.html')


@flask_app.route("/predict", methods=["POST", "GET"])
def predict():
    # grab user input from form
    headline_input = request.form.get('headline')
    content_input = request.form.get('article')

    print("finished grabbing user input")

    # getting and preprocessing user inputted article
    article = headline_input + content_input
    #print(type(article))
    article = [complete_preprocessing(article)]
    #print(type(article))
    emo_vector = emo_vectorizer.transform(article)
    #print("emo vector: ", emo_vector)
    fake_news_vector = fake_news_vectorizer.transform(article)
    #print("fake news vector: ", fake_news_vector)

    print("finished preprocessing user input")

    # predicting article's emotionality
    emotion_output = (emo_model.predict([emo_vector]))[0]
    #del emo_vector

    print("finished predicting emotionality")

    num_emo_dict = {
        "joy": (round((emotion_output[0]) * 100)),
        "sadness": (round((emotion_output[1]) * 100)),
        "anger": (round((emotion_output[2]) * 100)),
        "fear": (round((emotion_output[3]) * 100)),
        "surprise": (round((emotion_output[4]) * 100)),
    }

    emo_dict = {
        "joy": f"{(num_emo_dict['joy'])}%",
        "sadness": f"{(num_emo_dict['sadness'])}%",
        "anger": f"{(num_emo_dict['anger'])}%",
        "fear": f"{(num_emo_dict['fear'])}%",
        "surprise": f"{(num_emo_dict['surprise'])}%",
    }

    dominant_emotion = max(num_emo_dict, key=num_emo_dict.get)
    #del num_emo_dict

    # predicting article's credibility
    credibility_output = (fake_news_model.predict([fake_news_vector]))[0]

    print("finished predicting credibility")

    num_credibility_dict = {
        "fake news": (round((credibility_output[1]) * 100)),
        "real news": (round((credibility_output[0]) * 100))
    }
    credibility_dict = {
        "fake news": f"{(num_credibility_dict['fake news'])}%",
        "real news": f"{(num_credibility_dict['real news'])}%"
    }
    article_credibility = max(num_credibility_dict, key=num_credibility_dict.get)

    # formatting data to be sent to html
    emotion_result = f"The dominant emotion in the article is {dominant_emotion}. The probability distribution is {emo_dict}"
    credibility_result = f"The article is likely to be {article_credibility}. The probability distribution is {credibility_dict}"

    headline_placeholder = headline_input
    article_placeholder = content_input

    data = {'fake': int(num_credibility_dict["fake news"]),
            'real': int(num_credibility_dict["real news"]),
            'joy': int(num_emo_dict['joy']),
            'sadness': int(num_emo_dict['sadness']),
            'anger': int(num_emo_dict['anger']),
            'fear': int(num_emo_dict['fear']),
            'surprise': int(num_emo_dict['surprise'])
            }

    print("finished formatting output data")

    return render_template("index.html", emo_result=emotion_result, cred_result=credibility_result, chart_data=data,
                           headline_placeholder=headline_placeholder, article_placeholder=article_placeholder
                           )


if __name__ == "__main__":
    flask_app.run(debug=False)
