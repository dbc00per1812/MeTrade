import torch
import torch.nn as nn
import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from datetime import datetime

# Step 1: Download Stock Data
def download_stock_data(symbol):
    end_date = datetime.now()
    start_date = end_date.replace(year=end_date.year - 5)
    data = yf.download(symbol, start=start_date, end=end_date)
    return data

# Step 2: Preprocessing Data
def preprocess_data(data):
    close_prices = data['Close'].values.reshape(-1, 1)
    
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(close_prices)

    X = []
    y = []
    window_size = 60
    for i in range(window_size, len(scaled_data)):
        X.append(scaled_data[i - window_size:i, 0])
        y.append(scaled_data[i, 0])

    X = np.array(X)
    y = np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    return train_test_split(X, y, test_size=0.2), scaler

# Step 3: Define the Neural Network Model
class StockPredictor(nn.Module):
    def __init__(self):
        super(StockPredictor, self).__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=50, num_layers=2, batch_first=True)
        self.fc = nn.Linear(50, 1)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:, -1, :]
        out = self.fc(lstm_out)
        return out

# Step 4: Train the Model
def train_model(model, X_train, y_train, epochs=100, batch_size=64, lr=0.001):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(X_train_tensor)
        loss = criterion(output, y_train_tensor.unsqueeze(1))
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.4f}')
    
    print("Training complete")

# Step 5: Make Predictions and Give Buy/Sell/Hold Recommendation
def make_predictions(model, X_test, scaler, actual_price):
    model.eval()
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    predictions = model(X_test_tensor).detach().numpy()
    predictions = scaler.inverse_transform(predictions)

    last_predicted_price = predictions[-1][0]
    
    # Get the current price
    current_price = actual_price[-1]
    
    # Recommendation Logic
    if last_predicted_price > current_price * 1.02:  # If predicted price is 2% higher
        return "Buy", predictions
    elif last_predicted_price < current_price * 0.98:  # If predicted price is 2% lower
        return "Sell", predictions
    else:
        return "Hold", predictions

# Step 6: Plot Predictions vs Actual Prices
def plot_predictions(predictions, actual_prices):
    plt.figure(figsize=(10, 6))
    plt.plot(actual_prices, label="Actual Prices")
    plt.plot(predictions, label="Predicted Prices")
    plt.legend()
    plt.show()

# Main Function
if __name__ == "__main__":
    # Prompt the user for the stock ticker
    stock_ticker = input("Enter the stock ticker symbol: ")
    
    # Download and preprocess data
    data = download_stock_data(stock_ticker)
    (X_train, X_test, y_train, y_test), scaler = preprocess_data(data)
    
    # Define and train the model
    model = StockPredictor()
    train_model(model, X_train, y_train, epochs=50)
    
    # Make predictions and get recommendation
    recommendation, predictions = make_predictions(model, X_test, scaler, data['Close'].values)
    
    # Rescale actual prices for comparison
    actual_prices = scaler.inverse_transform(y_test.reshape(-1, 1))
    
    # Display recommendation
    print(f"Recommendation for {stock_ticker}: {recommendation}")
    
    # Plot the results
    plot_predictions(predictions, actual_prices)
