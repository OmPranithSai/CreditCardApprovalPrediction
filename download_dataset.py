"""
download_dataset.py
Helper script to download the Credit Card Fraud Detection dataset from Kaggle.

Prerequisites:
1. Kaggle account
2. kaggle API key (kaggle.json in ~/.kaggle/)

Steps:
1. Go to https://www.kaggle.com/settings -> Create New API Token
2. Place kaggle.json in C:\\Users\\<username>\\.kaggle\\
3. Run: pip install kaggle
4. Run: python download_dataset.py
"""

import os
import subprocess
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, 'dataset')


def download():
    os.makedirs(DATASET_DIR, exist_ok=True)
    print("Downloading Credit Card Fraud Detection dataset from Kaggle...")

    try:
        subprocess.run([
            sys.executable, '-m', 'kaggle', 'datasets', 'download',
            '-d', 'mlg-ulb/creditcardfraud',
            '-p', DATASET_DIR, '--unzip'
        ], check=True)
        print(f"\nDataset downloaded and extracted to: {DATASET_DIR}")
        print("File: creditcard.csv")

    except subprocess.CalledProcessError:
        print("\nKaggle download failed. Manual download required.")
        print("\nAlternative: Download manually from:")
        print("  https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud")
        print(f"  Place creditcard.csv in: {DATASET_DIR}")
    except FileNotFoundError:
        print("\nKaggle CLI not found. Install with: pip install kaggle")
        print("Then run this script again or download manually.")


if __name__ == '__main__':
    download()
