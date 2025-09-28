print("Importing dependencies...")
from flask import Flask, request, render_template, flash, redirect, url_for
from emowarn_utils import *
print("Finished importing dependencies")

print("Importing models...")
emo_model, emo_vectorizer, fake_news_model, fake_news_vectorizer = load_models_and_vectorizers()
print("finished loading ai models")
# Create flask app
flask_app = Flask(__name__)
flask_app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'

@flask_app.route("/", methods=['GET'])
def home(): #headline_placeholder = "Enter news headline...", article_placeholder = "Enter news article..."):
    data = {'fake': 50,
            'real': 50,
            'joy': 20,
            'sadness': 20,
            'anger': 20,
            'fear': 20,
            'surprise': 20
            }

    headline_placeholder = "Enter news headline..."
    article_placeholder = "Enter news article..."
    return render_template("index.html", chart_data=data,
                           headline_placeholder=headline_placeholder,
                           article_placeholder=article_placeholder)


@flask_app.route('/instructions', methods=['GET'])
def instructions():
    return render_template('instructions.html')


@flask_app.route("/predict", methods=["POST", "GET"])
def predict():
    #grab user input from form
    headline_input = request.form.get('headline')
    content_input = request.form.get('article')

    if headline_input == "" or content_input == "":
        flash("Incomplete input! Please enter the complete article and its headline.", "error")
        #if headline_input == "": flash("Article headline")
        #if content_input == "": flash("Article main body")
        return redirect(url_for('home'))

    article = headline_input + " " + content_input
    print("finished grabbing user input")

    #preprocessing user inputted article
    emo_vector = emo_vectorizer.transform([article])
    print("emotion vector: ", emo_vector)

    fake_news_vector = fake_news_vectorizer.transform([article])
    print("finished preprocessing user input")

    #predicting article's emotionality
    emotion_output = emo_model.predict_proba(emo_vector[0])
    emotion_output = emotion_output[0].tolist()

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

    #predicting article's credibility
    credibility_output = fake_news_model.predict_proba(fake_news_vector[0])
    credibility_output = credibility_output[0].tolist()

    num_credibility_dict = {
        "fake news": (round((credibility_output[1]) * 100)),
        "real news": (round((credibility_output[0]) * 100))
    }

    credibility_dict = {
        "fake news": f"{(num_credibility_dict['fake news'])}%",
        "real news": f"{(num_credibility_dict['real news'])}%"
    }

    article_credibility = max(num_credibility_dict, key=num_credibility_dict.get)

    #formatting data to be sent to html
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
    flask_app.run(debug=False, host = "127.0.0.128", port = 5000)
