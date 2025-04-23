import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import StandardScaler
from neuralforecast.core import NeuralForecast
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error


save_path = 'C:/Users/franc/PycharmProjects/TFC/checkpoints/lstm_test_run'
scaler_path = 'C:/Users/franc/PycharmProjects/TFC/checkpoints/lstm_scaler.pkl'
file_path = 'BitcoinPrice.csv'

# Carregar dados reais
full_df = pd.read_csv(file_path)
full_df.rename(columns={'Date': 'ds', 'Low': 'y', 'Currency': 'unique_id'}, inplace=True)
full_df['ds'] = pd.to_datetime(full_df['ds'], errors='coerce')
full_df = full_df.dropna(subset=['ds'])
full_df['unique_id'] = full_df['unique_id'].astype(str).fillna("BTC").str.strip()
full_df['y'] = pd.to_numeric(full_df['y'], errors='coerce')
full_df = full_df.dropna(subset=['y'])
full_df = full_df.sort_values(by='ds').reset_index(drop=True)

# Carregar scaler
with open(scaler_path, 'rb') as f:
    scaler = pickle.load(f)

# Carregar modelo treinado e fazer previsão
nf_loaded = NeuralForecast.load(path=save_path)
Y_hat_df = nf_loaded.predict()
y_pred = Y_hat_df['LSTM-median'].to_numpy().reshape(-1, 1)
y_pred_original = scaler.inverse_transform(y_pred).flatten()

# Obter data final do treino (30 nov 2020) e construir datas futuras
last_train_date = pd.to_datetime('2020-11-30')
future_dates = pd.date_range(start=last_train_date + pd.Timedelta(days=1), periods=len(y_pred_original), freq='D')

# Construir DataFrame de previsão
df_pred = pd.DataFrame({'ds': future_dates, 'y_pred': y_pred_original})
df_dez = full_df[(full_df['ds'] >= '2020-12-01') & (full_df['ds'] <= '2020-12-31')]
y_true_original = df_dez['y'].to_numpy()
y_true_original = y_true_original[:len(y_pred_original)]
# Guardar previsões
results_dir = 'C:/Users/franc/PycharmProjects/TFC/results'
os.makedirs(results_dir, exist_ok=True)
output_df = df_pred.copy()
output_df.to_csv(f'{results_dir}/lstm_dez2020_predictions.csv', index=False)

# Gráfico: dados reais completos + previsão a partir de dezembro
fig, ax = plt.subplots(figsize=(20, 7))

# Parte real
df_real_plot = full_df[full_df['ds'] <= last_train_date]
plt.plot(df_real_plot['ds'], df_real_plot['y'], label='Dados Reais (até Nov 2020)', color='black', linewidth=2)

# Previsão
plt.plot(df_pred['ds'], df_pred['y_pred'], label='Previsão (Dez 2020)', color='blue', linestyle='dashed', linewidth=2)

# Se quiseres sobrepor os dados reais de dezembro (em cinza)
df_dez = full_df[(full_df['ds'] >= pd.to_datetime('2020-12-01')) & (full_df['ds'] <= pd.to_datetime('2020-12-31'))]
if not df_dez.empty:
    plt.plot(df_dez['ds'], df_dez['y'], label='Dados Reais (Dez 2020)', color='gray', linewidth=2, alpha=0.7)

def huber_mq_loss(y_true, y_pred, delta=0.5):
    error = y_true - y_pred
    return np.where(np.abs(error) < delta,
                    0.5 * error**2,
                    delta * (np.abs(error) - 0.5 * delta)).mean()

metrics = {
    'Modelo': 'LSTM',
    'MAE': mean_absolute_error(y_true_original, y_pred_original),
    'MAPE': mean_absolute_percentage_error(y_true_original, y_pred_original),
    'RMSE': mean_squared_error(y_true_original, y_pred_original, squared=False),
    'HuberMQLoss': huber_mq_loss(y_true_original, y_pred_original)
}

pd.DataFrame([metrics]).to_csv(f'{results_dir}/metrics_lstm.csv', index=False)


ax.set_title('Previsão com LSTM e Dados Reais', fontsize=22)
ax.set_ylabel('Preço Bitcoin', fontsize=20)
ax.set_xlabel('Data', fontsize=20)
ax.legend(prop={'size': 15})
ax.grid()
plt.show()
