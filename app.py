from flask import Flask, request, jsonify
import pandas as pd
from analyze_players import train_model, load_data, preprocess_data

app = Flask(__name__)

# Load and preprocess data once
df = load_data('player_data.csv')
df = preprocess_data(df)
model = train_model(df[['age', 'games_played', 'score']], df['points'])

@app.route('/predict', methods=['POST'])
def predict():
    content = request.json
    data = pd.DataFrame([content])
    prediction = model.predict(data)
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True)