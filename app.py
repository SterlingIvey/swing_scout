from flask import Flask, request, jsonify, abort
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
    # Validate the input
    required_columns = ['age', 'games_played', 'score']
    if not all(col in content for col in required_columns):
        return jsonify({'error': 'Missing data for required fields'}), 400
    
    try:
        data = pd.DataFrame([content])
        prediction = model.predict(data)
        return jsonify({'prediction': prediction.tolist()})
    except Exception as e:
        abort(400, description=str(e))

if __name__ == '__main__':
    app.run(debug=True)