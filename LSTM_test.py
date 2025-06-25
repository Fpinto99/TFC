import matplotlib
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')
plt.rcParams.update({'font.size': 12})
import os
from sklearn.preprocessing import StandardScaler
from neuralforecast.core import NeuralForecast
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

# Caminhos
scaler_path = 'C:/Users/franc/PycharmProjects/TFC_1/checkpoints/lstm_scaler.pkl'
file_path = 'BTC_2018_2024.csv'
results_dir = 'C:/Users/franc/PycharmProjects/TFC_1/checkpoints/lstm_test_run'
os.makedirs(results_dir, exist_ok=True)

# Carregar dados reais
full_df = pd.read_csv(file_path)
full_df.rename(columns={'Date': 'ds', 'Low': 'y', 'currency': 'unique_id'}, inplace=True)
full_df['ds'] = pd.to_datetime(full_df['ds'], errors='coerce')
full_df = full_df.dropna(subset=['ds'])
full_df['unique_id'] = full_df['unique_id'].astype(str).fillna("BTC").str.strip()
full_df['y'] = pd.to_numeric(full_df['y'], errors='coerce')
full_df = full_df.dropna(subset=['y'])
full_df = full_df.sort_values(by='ds').reset_index(drop=True)

# Carregar scaler
with open(scaler_path, 'rb') as f:
    scaler = pickle.load(f)

# Data final do treino e datas futuras
last_train_date = pd.to_datetime('2023-11-30')
future_dates = pd.date_range(start=last_train_date + pd.Timedelta(days=1), periods=31, freq='D')

# Dados reais de dezembro
df_dez = full_df[(full_df['ds'] >= '2023-12-01') & (full_df['ds'] <= '2023-12-31')]
y_true_original = df_dez['y'].to_numpy()


# Função para calcular HuberMQ
def huber_mq_loss(y_true, y_pred, delta=0.5):
    error = y_true - y_pred
    return np.where(np.abs(error) < delta,
                    0.5 * error ** 2,
                    delta * (np.abs(error) - 0.5 * delta)).mean()


# Modelos a comparar
modelos = {
    'Default': 'C:/Users/franc/PycharmProjects/TFC_1/checkpoints/lstm_test_run/default',
    'Modificado': 'C:/Users/franc/PycharmProjects/TFC_1/checkpoints/lstm_test_run/moded'
}

# Gráfico
fig, ax = plt.subplots(figsize=(20, 7))
df_real_plot = full_df[full_df['ds'] <= last_train_date]
plt.plot(df_real_plot['ds'], df_real_plot['y'], label='Dados Reais (até Nov 2023)', color='black', linewidth=2)

# Cores para cada modelo
colors = {
    'Default': 'blue',
    'Modificado': 'green'
}

# Avaliação e previsão
for nome, path in modelos.items():
    nf_loaded = NeuralForecast.load(path=path)
    Y_hat_df = nf_loaded.predict()
    y_pred = Y_hat_df['LSTM-median'].to_numpy().reshape(-1, 1)
    y_pred_original = scaler.inverse_transform(y_pred).flatten()

    # Truncar para evitar mismatch
    y_true_truncado = y_true_original[:len(y_pred_original)]

    # Guardar previsões
    df_pred = pd.DataFrame({'ds': future_dates[:len(y_pred_original)], f'y_pred_{nome}': y_pred_original})
    df_pred.to_csv(f'{results_dir}/pred_{nome}.csv', index=False)

    # Plotar
    plt.plot(df_pred['ds'], df_pred[f'y_pred_{nome}'], label=f'Previsão {nome}', linestyle='dashed', linewidth=2,
             color=colors[nome])

    # Métricas
    metrics = {
        'Modelo': nome,
        'MAE': mean_absolute_error(y_true_truncado, y_pred_original),
        'MAPE': mean_absolute_percentage_error(y_true_truncado, y_pred_original),
        'RMSE': mean_squared_error(y_true_truncado, y_pred_original) ** 0.5,
        'HuberMQLoss': huber_mq_loss(y_true_truncado, y_pred_original)
    }
    pd.DataFrame([metrics]).to_csv(f'{results_dir}/metrics_{nome}.csv', index=False)

# Sobrepor os dados reais de dezembro
if not df_dez.empty:
    plt.plot(df_dez['ds'], df_dez['y'], label='Dados Reais (Dez 2023)', color='gray', linewidth=2, alpha=0.7)

ax.set_title('Comparação de Previsões LSTM', fontsize=22)
ax.set_ylabel('Preço Bitcoin', fontsize=20)
plt.xlim(pd.Timestamp('2023-09-01'))
ax.set_xlabel('Data', fontsize=20)
ax.legend(prop={'size': 15})
ax.grid()
plt.show()
