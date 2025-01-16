from flask import Flask, render_template, request
from pipeline.prediction_pipeline import hybrid_recommendation

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    recommendations = None  # Initialize recommendations as None

    if request.method == 'POST':
        try:
            # Extract the user_id from the form
            user_id = int(request.form['userId'])  # Convert to int for processing
            
            # Call the hybrid recommendation function
            recommendations = hybrid_recommendation(user_id, user_weight=0.6, content_weight=0.4)
        except Exception as e:
            recommendations = [f"An error occurred: {e}"]

    return render_template('index.html', recommendations=recommendations)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
