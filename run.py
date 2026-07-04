"""
run.py
Convenience script to run the Flask application.
Usage: python run.py
"""

import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'best_model.pkl')

if not os.path.exists(MODEL_PATH):
    print("=" * 60)
    print("MODEL NOT FOUND")
    print(f"Expected: {MODEL_PATH}")
    print("\nPlease run train_model.py first:")
    print("  python train_model.py")
    print("=" * 60)
    sys.exit(1)

from app import app

if __name__ == '__main__':
    print("Starting Credit Card Approval Prediction System...")
    print("Access the application at: http://127.0.0.1:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
