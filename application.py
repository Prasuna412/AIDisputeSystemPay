from flask import Flask, request, render_template, redirect, flash, url_for, session
import numpy as np
import pandas as pd
from joblib import load

from sklearn.preprocessing import MinMaxScaler
from src.components.preprocessor import TextCleaner, TextVectorizer
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
application.secret_key = 'your_secret_key'  # Needed for session management

# Admin credentials (use environment variables or a secure method in production)
ADMIN_USERNAME = 'MidPay.ai'
ADMIN_PASSWORD = 'MidPay2024!'

# Route for home page


@application.route('/', methods=['GET'])
def home():
    if 'logged_in' not in session or not session['logged_in']:
        return redirect(url_for('admin_login'))
    return render_template('index.html')

# Route for admin login


@application.route('/login', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
            session['logged_in'] = True
            flash('Login successful!')
            # Ensure this redirects to an existing route
            return redirect(url_for('home'))
        else:
            flash('Invalid credentials. Please try again.')
    return render_template('admin_login.html')

# Route for logout


@application.route('/logout', methods=['POST'])
def logout():
    session.pop('logged_in', None)
    flash('You have been logged out.')
    return redirect(url_for('admin_login'))

# Route for severity prediction


@application.route('/prediction', methods=['GET', 'POST'])
def predict_severity_result():
    if 'logged_in' not in session or not session['logged_in']:
        # Redirect to login if not logged in
        return redirect(url_for('admin_login'))
    if request.method == 'GET':
        return render_template('severity.html', prediction=None, prob_high=None, prob_medium=None, prob_low=None, error=None)
    else:
        try:
            data = CustomData(text=request.form.get('text'))
            pred_df = data.get_data()

            predict_pipeline = PredictPipeline()
            results, probabilities = predict_pipeline.predict(pred_df)
            prob_high = probabilities[0][2]
            prob_medium = probabilities[0][1]
            prob_low = probabilities[0][0]

            return render_template('severity.html', prediction=results[0].item(), prob_high=prob_high, prob_medium=prob_medium, prob_low=prob_low, error=None)
        except Exception as e:
            return render_template('severity.html', prediction=None, prob_high=None, prob_medium=None, prob_low=None, error=str(e))


if __name__ == '__main__':
    application.run(debug=True)
