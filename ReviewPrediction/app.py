from flask import Flask, render_template, request, redirect, url_for
import joblib, re
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from preprocessor import preprocessor

app = Flask(__name__)

##################################
@app.route('/')
def home():
    return render_template("home.html")

@app.route('/prediction', methods = ['get', 'post'])
def prediction():
    if request.method == "POST":
        review = request.form.get("Review")
        if re.findall(r"^[-\d]",review):
            prediction = 0
            return render_template("output.html", prediction = prediction, review = review)

        elif re.findall(r"^[\W_]",review) or review == "":
            prediction = "Enter valid review..."
            return render_template("home.html", prediction = prediction)        
        
        
        elif re.findall(r"^[^\d]+$",review):
            preprocessor_review = preprocessor(review)
            model = joblib.load("C:\\Users\\Suswarah\\OneDrive\\Desktop\\flask\\Flipkart_ReviewPrediction\\Model\\logstic_model.pkl")
            prediction = model.predict([preprocessor_review])
            prediction = int(prediction[0])

            return render_template("output.html", prediction = prediction,  review = review)

        elif  int(review) >= 3:
            prediction = 1
            return render_template("output.html", prediction = prediction,  review = review)
    elif request.method == "GET":
        return render_template("home.html")
    else:
        return redirect(url_for('prediction'))

#################################

if __name__ == "__main__":
    app.run(debug = True, host = "0.0.0.0")

