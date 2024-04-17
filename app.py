from flask import Flask, request, jsonify, abort
import pandas as pd
import logging
from analyze_players import train_model, load_data, preprocess_data

app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG)

# Load and preprocess data once
try:
    df = load_data('player_data.csv')
    df = preprocess_data(df)
    model = train_model(df[['age', 'games_played', 'score']], df['points'])
except Exception as e:
    app.logger.error(f"Failed to load or preprocess data: {e}")
    raise RuntimeError(f"Data loading failed: {e}")

@app.route('/predict', methods=['POST'])
def predict():
    content = request.json
    required_columns = ['age', 'games_played', 'score']
    if not all(col in content for col in required_columns):
        app.logger.error("Missing data for required fields")
        return jsonify({'error': 'Missing data for required fields'}), 400
    
    try:
        data = pd.DataFrame([content])
        prediction = model.predict(data)
        return jsonify({'prediction': prediction.tolist()})
    except Exception as e:
        app.logger.error(f"Prediction failed: {e}")
        abort(400, description=str(e))

if __name__ == '__main__':
    app.run(debug=False)  # Turn off debug mode for production