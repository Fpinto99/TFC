import torch
import numpy as np
import pandas as pd
import time
import pickle
from sklearn.preprocessing import StandardScaler
from neuralforecast.models import RNN
from neuralforecast.core import NeuralForecast
from neuralforecast.losses.pytorch import HuberMQLoss

def load_data(file_path):
    df = pd.read_csv(file_path)
    df.rename(columns={'Date': 'ds', 'Low': 'y', 'currency': 'unique_id'}, inplace=True)
    df['ds'] = pd.to_datetime(df['ds'], format='%Y-%m-%d', errors='coerce')
    df = df.dropna(subset=['ds'])

    df['unique_id'] = df['unique_id'].astype(str).fillna("BTC").str.strip()
    df['y'] = pd.to_numeric(df['y'], errors='coerce')
    df = df.dropna(subset=['y']).sort_values(by=['unique_id', 'ds']).reset_index(drop=True)

    # Cortar dados até 30 de novembro de 2023
    corte_data = pd.to_datetime('2023-11-30')
    df = df[df['ds'] <= corte_data]

    # Normaliza e guarda o scaler
    scaler = StandardScaler()
    df['y'] = scaler.fit_transform(df[['y']])

    scaler_path = 'C:/Users/franc/PycharmProjects/TFC_1/checkpoints/rnn_scaler.pkl'
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)

    return df

def train_model(df, horizon, save_path,params):
    start_time = time.time()

    model = RNN(
        h=horizon,
        loss=HuberMQLoss(),
        scaler_type='standard',
        **params
    )

    nf = NeuralForecast(models=[model], freq='D')
    nf.fit(df=df)
    nf.save(path=save_path, model_index=None, overwrite=True, save_dataset=True)

    print(f"Treino concluído em {time.time() - start_time:.2f} segundos!")

if __name__ == "__main__":
    file_path = 'BTC_2018_2024.csv'
    df = load_data(file_path)
    horizon = 31

    # Modelo 1 - Default
    params_1 = {
        'input_size' : 168,
        'max_steps': 1000,
        'learning_rate': 1e-4,
        'windows_batch_size': 32,
        'val_check_steps': 100,
        'random_seed': 42,
    }
    train_model(df, horizon, 'C:/Users/franc/PycharmProjects/TFC_1/checkpoints/rnn_test_run/default', params_1)

    # Modelo 2 - Alterado
    params_2 = {
        'input_size' : 256,
        'max_steps': 3000,
        'learning_rate': 1e-4,
        'windows_batch_size': 64,
        'val_check_steps': 100,
        'random_seed': 42,
    }
    train_model(df, horizon, 'C:/Users/franc/PycharmProjects/TFC_1/checkpoints/rnn_test_run/moded', params_2)

