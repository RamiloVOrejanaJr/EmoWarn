from flask import Flask, request, render_template
from nltk.corpus import stopwords
import tf_keras
import re


#define preprocessing method
def complete_preprocessing(article):
    article = str(article).lower()
    article = re.sub('[^a-zA-Z]', ' ', article)
    article = re.sub('\s+[^a-zA-Z]\s+', ' ', article)
    pattern = re.compile(r'\b(' + r'|'.join(stopwords.words('english')) + r')\b\s*')
    article = pattern.sub('', article)
    return article


#get emotion clasification model
emo_model = tf_keras.models.load_model('saved_model/my_model')

fake_news_model = tf_keras.models.load_model('fake_news_model/my_model')

# Create flask app
flask_app = Flask(__name__)


@flask_app.route("/")
def home():
    return render_template("index.html")

@flask_app.route('/instructions') 
def instructions(): 
    return render_template('instructions.html')

@flask_app.route("/predict", methods=["POST", "GET"])
def predict():
    #grab user input from form
    headline_input = request.form.get('headline')
    content_input = request.form.get('article')

    #getting and preprocessing user inputted article
    article = headline_input + content_input
    article = complete_preprocessing(article)

    #predicting article's emotionality
    emotion_output = (emo_model.predict([article]))[0]
    emo_dict = {
        "joy": f"{(round((emotion_output[0])*100))}%",
        "sadness": f"{(round((emotion_output[1])*100))}%",
        "anger": f"{(round((emotion_output[2])*100))}%",
        "fear": f"{(round((emotion_output[3])*100))}%",
        "surprise": f"{(round((emotion_output[4])*100))}%"
    }
    dominant_emotion = max(emo_dict, key=emo_dict.get)

    #predicting article's credibility
    credibility_output = (fake_news_model.predict([article]))[0]
    credibility_dict = {
        "fake news": f"{(round((credibility_output[0])*100))}%",
        "real news": f"{(round((credibility_output[1])*100))}%",
    }
    article_credibility = max(credibility_dict, key=credibility_dict.get)

    #formatting data to be sent to html
    emotion_result = f"The dominant emotion is {dominant_emotion}. The probability distribution is {emo_dict}"
    credibility_result = f"The article with its headline is likely to be {article_credibility}. The probability distribution is {credibility_dict}"

    print(f'Emotion output: {emotion_output}')
    print(f'Credibility output: {credibility_output}')
    print(type(emotion_output))

    print(f'{headline_input} OR {content_input}')
    return render_template("index.html", emo_result=emotion_result, cred_result=credibility_result,
                           emo_array=emotion_output, cred_array=credibility_output)
    

if __name__ == "__main__":
    flask_app.run(debug=True)
