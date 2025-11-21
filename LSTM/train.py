import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

from LSTM import TimeSeriesDataset, LSTMModel

teams = ['100_Thieves', 'Bilibili_Gaming', 'Dplus_KIA', 'FlyQuest', 'Fnatic', 'Fukuoka_SoftBank_HAWKS_gaming',
         'G2_Esports', 'GAM_Esports', 'Gen.G', 'Hanwha_Life_Esports', 'LNG_Esports', 'MAD_Lions_KOI', 
         'Movistar_R7', 'paiN_Gaming', 'PSG_Talon', 'T1', 'Team_Liquid', 'Top_Esports', 'Vikings_Esports', 
         'Weibo_Gaming']

data_dir = os.path.join('LSTM', 'positive_data')

save_dir = os.path.join('LSTM', 'results_positive')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

checkpoints_dir = os.path.join('LSTM', 'checkpoints_positive')
if not os.path.exists(checkpoints_dir):
    os.makedirs(checkpoints_dir)

predictions_file = 'LSTM/2024_worlds_prediction_positive.txt'
with open(predictions_file, 'w') as f:
    pass

evaluations_metrics_file = 'LSTM/evaluation_metrics_positive.txt'
with open(evaluations_metrics_file, 'w') as f:
    f.write('Team, mean_squared_error,mean_absolute_error, data_range, r2, mean_absolute_percentage_error\n')

# Define device to train on (CPU/GPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Training on', device)

for team_name in teams:
    print('----------------------------------------------------')
    print('Training model for', team_name)

    file = os.path.join(data_dir, f'{team_name}.csv')

    data = pd.read_csv(file)

    # Sort data by date to maintain temporal order and reset index
    data.sort_values('Date', inplace=True)
    data.reset_index(drop=True, inplace=True)

    # Feature scaling
    scaler = MinMaxScaler(feature_range=(0, 1))
    data['scaled_score'] = scaler.fit_transform(data[['Score']])

    # Create sequences for LSTM
    def create_sequences(data, seq_length):
        sequences = []
        targets = []
        for i in range(len(data) - seq_length):
            seq = data['scaled_score'].iloc[i:i+seq_length].values
            label = data['scaled_score'].iloc[i+seq_length]
            sequences.append(seq)
            targets.append(label)
        return np.array(sequences), np.array(targets)

    seq_length = 3  # HYPERPARAMETER: Number of previous steps to use for prediction
    X, y = create_sequences(data, seq_length)

    # Split dataset
    train_size = int(len(X) * 0.8)
    X_train = X[:train_size]
    X_test = X[train_size:]
    y_train = y[:train_size]
    y_test = y[train_size:]

    # Create dataloaders
    batch_size = 32

    train_dataset = TimeSeriesDataset(X_train, y_train)
    test_dataset = TimeSeriesDataset(X_test, y_test)
    all_dataset = TimeSeriesDataset(X, y)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    all_loader = DataLoader(all_dataset, batch_size=batch_size)

    # Initialize model, loss function, and optimizer
    model = LSTMModel(input_size=1, hidden_size=50, num_layers=3) # HYPERPARAMETERS: hidden_size, num_layers
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    model.to(device)

    # Train
    num_epochs = 300 # HYPERPARAMETER

    for epoch in range(num_epochs):
        model.train()
        for sequences, targets in train_loader:
            sequences = sequences.unsqueeze(-1)  # Add input_size dimension
            sequences, targets = sequences.to(device), targets.to(device)

            # Forward pass
            outputs = model(sequences)
            loss = criterion(outputs, targets.unsqueeze(-1))

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for sequences, targets in test_loader:
                sequences = sequences.to(device)
                targets = targets.to(device)
                sequences = sequences.unsqueeze(-1)
                outputs = model(sequences)
                loss = criterion(outputs, targets.unsqueeze(-1))
                val_losses.append(loss.item())

        print(f'Epoch [{epoch+1}/{num_epochs}], '
            f'Training Loss: {loss.item():.4f}, '
            f'Validation Loss: {np.mean(val_losses):.4f}')
        
    # Evaluate the model
    model.eval()
    predictions = []
    actuals = []

    with torch.no_grad():
        for sequences, targets in all_loader:#test_loader:
            sequences = sequences.unsqueeze(-1)

            sequences, targets = sequences.to(device), targets.to(device)

            outputs = model(sequences)
            outputs = outputs.view(-1)
            targets = targets.view(-1)

            predictions.extend(outputs.cpu().tolist())
            actuals.extend(targets.cpu().tolist())

    # Inverse transform to get actual scores
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    actuals = scaler.inverse_transform(np.array(actuals).reshape(-1, 1))

    # Calculate evaluation metrics
    from sklearn.metrics import mean_squared_error, mean_absolute_error\

    mse = mean_squared_error(actuals, predictions)
    mae = mean_absolute_error(actuals, predictions)
    print(f'MSE: {mse:.4f}, MAE: {mae:.4f}')

    # Find scale of the error for reference
    scale = data['Score'].max() - data['Score'].min()
    print(f'Scale of the absolute error: {scale:.4f}')

    # Print more model evaluation metrics
    from sklearn.metrics import r2_score

    r2 = r2_score(actuals, predictions)
    print(f'R^2: {r2:.4f}')

    from sklearn.metrics import mean_absolute_percentage_error

    mape = mean_absolute_percentage_error(actuals, predictions)
    print(f'MAPE: {mape:.4f}')

    with open(evaluations_metrics_file, 'a') as f:
        f.write(f'{team_name}, {mse:.4f}, {mae:.4f}, {scale:4f}, {r2:.4f}, {mape:.4f}\n')

    # Visualize results
    plt.figure(figsize=(12,6))
    plt.plot(actuals, label='Actual')
    plt.plot(predictions, label='Predicted')
    plt.xlabel('Sample')
    plt.ylabel('Team Performance Score')
    plt.title(f'{team_name} Performance Score Prediction')
    plt.legend()

    # # Plot vertical lines for season start dates
    data['Date'] = pd.to_datetime(data['Date'])
    # spring_start = pd.to_datetime('2024-03-09')
    # msi_start = pd.to_datetime('2024-05-01')
    # summer_start = pd.to_datetime('2024-06-08')

    # spring_idx = data[data['Date'] == spring_start].index
    # msi_idx = data[data['Date'] == msi_start].index
    # summer_idx = data[data['Date'] == summer_start].index

    def get_nearest_index(date_series, target_date):
        nearest_date = date_series.iloc[(date_series - target_date).abs().argsort()[:1]]
        return nearest_date.index[0]

    # spring_idx = get_nearest_index(data['Date'], spring_start)
    # msi_idx = get_nearest_index(data['Date'], msi_start)
    # summer_idx = get_nearest_index(data['Date'], summer_start)

    # if spring_idx is not None:
    #     plt.axvline(spring_idx, color='r', linestyle='--', label='Spring Start')
    # if msi_idx is not None:
    #     plt.axvline(msi_idx, color='g', linestyle='--', label='MSI Start')
    # if summer_idx is not None:
    #     plt.axvline(summer_idx, color='b', linestyle='--', label='Summer Start')

    plt.savefig(os.path.join(save_dir, f'{team_name}_results.png'))
    #plt.show()

    # Save the model
    torch.save(model.state_dict(), os.path.join(checkpoints_dir, f'{team_name}lstm_model.pth'))

    # Predict the team performance score for worlds 9/28/2024
    worlds_date = pd.to_datetime('2024-10-14')
    worlds_idx = get_nearest_index(data['Date'], worlds_date)
    worlds_seq = data['scaled_score'].iloc[worlds_idx-seq_length:worlds_idx].values
    worlds_seq = torch.tensor(worlds_seq, dtype=torch.float32).unsqueeze(-1).to(device)
    worlds_seq = worlds_seq.unsqueeze(0)
    worlds_pred = model(worlds_seq)
    worlds_pred = scaler.inverse_transform(worlds_pred.squeeze().detach().cpu().numpy().reshape(-1, 1))
    print(f'Team performance score prediction for {team_name} at Worlds 2024: {worlds_pred[0][0]:.2f}')

    with open(predictions_file, 'a') as f:
        f.write(f'{team_name}, {worlds_pred[0][0]:.2f}\n')

# Load the model
# model = LSTMModel(input_size=1, hidden_size=50, num_layers=2)
# model.load_state_dict(torch.load('lstm_model.pth'))
# model.eval()
