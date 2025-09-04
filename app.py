from flask import Flask, request, render_template
from nltk.corpus import stopwords
import tf_keras
import re


# define preprocessing method
def complete_preprocessing(article):
    article = str(article).lower()
    article = re.sub('[^a-zA-Z]', ' ', article)
    article = re.sub('\s+[^a-zA-Z]\s+', ' ', article)
    pattern = re.compile(r'\b(' + r'|'.join(stopwords.words('english')) + r')\b\s*')
    article = pattern.sub('', article)
    return article


# get emotion clasification model
emo_model = tf_keras.models.load_model('saved_model/my_model')

fake_news_model = tf_keras.models.load_model('fake_news_model/my_model')

# Create flask app
flask_app = Flask(__name__)


@flask_app.route("/")
def home():
    data = {'fake': 50,
            'real': 50,
            'joy': 25,
            'sadness': 25,
            'anger': 25,
            'fear': 25,
            'surprise': 25
            }
    headline_placeholder = "Enter news headline..."
    article_placeholder = "Enter news article..."
    return render_template("index.html", chart_data=data,
                           headline_placeholder=headline_placeholder,
                           article_placeholder=article_placeholder)


@flask_app.route('/instructions')
def instructions():
    data = {'fake': 50,
            'real': 50,
            'joy': 25,
            'sadness': 25,
            'anger': 25,
            'fear': 25,
            'surprise': 25
            }
    return render_template('instructions.html')


@flask_app.route("/predict", methods=["POST", "GET"])
def predict():
    # grab user input from form
    headline_input = request.form.get('headline')
    content_input = request.form.get('article')

    # getting and preprocessing user inputted article
    article = headline_input + content_input
    article = complete_preprocessing(article)

    # predicting article's emotionality
    emotion_output = (emo_model.predict([article]))[0]

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

    # predicting article's credibility
    credibility_output = (fake_news_model.predict([article]))[0]
    num_credibility_dict = {
        "fake news": (round((credibility_output[0]) * 100)),
        "real news": (round((credibility_output[1]) * 100))
    }
    credibility_dict = {
        "fake news": f"{(num_credibility_dict['fake news'])}%",
        "real news": f"{(num_credibility_dict['real news'])}%"
    }
    article_credibility = max(num_credibility_dict, key=num_credibility_dict.get)

    # formatting data to be sent to html
    emotion_result = f"The dominant emotion in the article is {dominant_emotion}. The probability distribution is {emo_dict}"
    credibility_result = f"The article is likely to be {article_credibility}. The probability distribution is {credibility_dict}"

    print(f'Emotion output: {emotion_output}')
    print(f'Credibility output: {credibility_output}')
    print(type(emotion_output))

    print(f'{headline_input} OR {content_input}')

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

    return render_template("index.html", emo_result=emotion_result, cred_result=credibility_result, chart_data=data,
                           headline_placeholder=headline_placeholder, article_placeholder=article_placeholder
                           )


if __name__ == "__main__":
    flask_app.run(debug=True)
