EmoWarn is a light web application that uses a news article's headline and main content to analyze the 
probability of it being an article composed for the purpose of spreading misinformation, colloquially known
as fake news. It also analyzes the probability of which emotion the reader is likely to feel after reading
the article to attempt to warn the user about any emotional biases in the article. The emotions are based on 
Paul Ekman's basic emotions of Joy, Anger, Fear, Sadness, and Surprise. 

The app uses Python 3.10. As the web application has been built to be deployed on the free model of the 
pythonanywhere deployment service, the external package versions used are those used by default in 
pythonanywhere. More detailes can be found in their "Batteries Included" page.

The relevant external packages necessary to run the app:
1. numpy v1.21.6
2. flask v3.0.3
3. scikit-learn v1.0.2
