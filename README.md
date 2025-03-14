Load Forecasting using LSTM
Author: Lucinda McArthur

📌 Project Overview
This project explores deep learning techniques for energy load forecasting, focusing on LSTM (Long Short-Term Memory) and GRU (Gated Recurrent Units). The goal is to predict monthly energy costs using historical consumption data, weather conditions, and other relevant features.

📊 Key Features
✔️ Time-series forecasting using LSTM & GRU networks
✔️ Data preprocessing and feature engineering for better model accuracy
✔️ Model evaluation with Mean Absolute Error (MAE), Mean Squared Error (MSE), and R-squared
✔️ Hyperparameter tuning to optimize performance 

📂 Repository Contents
📜 Full Report: Detailed analysis, methodology, and findings
📊 Best-Performing Model: The final model architecture and evaluation
💻 Raw Script: Python code with full preprocessing, model training, and evaluation
⚙️ Installation & Setup
To run the project locally, follow these steps:

1️⃣ Clone the repository:
bash
Copy
Edit
git clone https://github.com/your-username/load-forecasting-lstm.git
cd load-forecasting-lstm

2️⃣ Install dependencies:
bash
Copy
Edit
pip install -r requirements.txt

3️⃣ Run the model:
bash
Copy
Edit
python main.py

📈 Results & Insights
The best-performing model was a three-layer LSTM architecture, showing improved forecasting accuracy.
Feature engineering (e.g., time-based & lagged features) contributed to slight performance improvements.
The final model achieved an MAE of 111.21, showing a slight improvement over the baseline model.
🚀 Future Work
Experiment with attention mechanisms for better sequence learning.
Incorporate additional external features like economic indicators.
Fine-tune dropout rates and learning rates for further optimization.
