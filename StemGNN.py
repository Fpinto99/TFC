import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time


from ray import tune
from neuralforecast.auto import AutoStemGNN
from neuralforecast.core import NeuralForecast
from neuralforecast.losses.pytorch import MAE
from neuralforecast.losses.pytorch import HuberMQLoss
from neuralforecast.losses.pytorch import MSE
from neuralforecast.losses.pytorch import MAPE
from numpy.lib.function_base import append

from statsmodels.tools.eval_measures import rmse #root mean squared error
from statsmodels.tools.eval_measures import rmspe #root mean squared percentage error
from statsmodels.tools.eval_measures import maxabs #maximum absolute error
from statsmodels.tools.eval_measures import meanabs #mean absolute error
from statsmodels.tools.eval_measures import medianabs #median absolute error
from statsmodels.tools.eval_measures import vare #variance of error
from statsmodels.tools.eval_measures import stde #standard deviation of error
from statsmodels.tools.eval_measures import iqr
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
torch.cuda.is_available()
import math

# Load the data
df = pd.read_csv('C:/Users/franc/OneDrive/Ambiente de Trabalho/BitcoinPrice.csv')
df.rename(columns={'Date': 'ds', 'Low':'y', 'Currency':'unique_id'}, inplace=True)

# Set the threshold until the max
#df['y'] = scaler.fit_transform(df[['y']])
max_index = df['y'].idxmax()
normalize_df = df.iloc[:max_index + 1]


# plt.figure(figsize=(6, 3))
# plt.plot(Y_df.y, 'k')
# plt.axis([0, len(list(Y_df.y)), 0, 100])
# plt.xlabel('Time (h)')
# plt.ylabel('Level (%)')
# plt.grid(linestyle='-', which='both')
# # plt.savefig('Singals_Ori.pdf', bbox_inches = 'tight')
# plt.show()

normalize_df['ds'] = pd.to_datetime(normalize_df['ds'], format='%Y-%m-%d')

horizon = 250

# Configuration of hyperparameter search space.
config = {
      "input_size": tune.choice([horizon]),
      "learning_rate": tune.loguniform(1e-4, 1e-1),
      "scaler_type": tune.choice(['robust', 'standard']),
      "max_steps": tune.choice([500, 1000]),
      "check_val_every_n_epoch": tune.choice([100]),
      "random_seed": tune.randint(1, 20),
}

start_time = time.time()
models = [AutoStemGNN(h=horizon,
                      n_series = 1,
                  loss=MAPE(),
                  config=config,
                  num_samples=1)]

nf = NeuralForecast(
    models=models,
    freq='D')


nf.fit(df=normalize_df)

save_path = 'C:/Users/franc/PycharmProjects/TFC/checkpoints4/test_run'

nf.save(path=save_path, model_index=None, overwrite=True, save_dataset=True)


# Recriar o NeuralForecast com o modelo carregado
model = AutoStemGNN(h=250, loss=MAPE(), config=config, num_samples=1)
nf_loaded = NeuralForecast(models=[model], freq='D')
nf_loaded = NeuralForecast.load(path='C:/Users/franc/PycharmProjects/TFC/checkpoints4/test_run')

# Agora pode prever
Y_hat_df = nf_loaded.predict(df=normalize_df)
y_pred = Y_hat_df['AutoStemGNN']
y_true = normalize_df[-horizon:].y
print(Y_hat_df.head())
end_time = time.time()
print('Time to create models:', end_time - start_time)

full_horizon = 250
n_predicts = math.ceil(full_horizon / model.h)
combined_train = normalize_df[['unique_id', 'ds', 'y']]
forecasts = []
for _ in range(n_predicts):
    step_forecast = nf_loaded.predict(df=combined_train)
    forecasts.append(step_forecast)
    step_forecast =  step_forecast.rename(columns={'AutoStemGNN': 'y'})
    # drop columns
    step_forecast = step_forecast[['ds', 'y']]
    step_forecast = step_forecast.reset_index()
    combined_train = pd.concat([combined_train, step_forecast])
    combined_train = combined_train.reset_index(drop=True)
a = pd.concat(forecasts)
print(f'{(rmse(y_true, y_pred)):.2E} & {(rmspe(y_true, y_pred)):.2E} & {(maxabs(y_true, y_pred)):.2E} & {(meanabs(y_true, y_pred)):.2E} & {(medianabs(y_true, y_pred)):.2E} ')

fig, ax = plt.subplots(1, 1, figsize = (20, 7))
plot_df = pd.concat([normalize_df, a]).set_index('ds')
plot_df[['y', 'AutoStemGNN']].plot(ax=ax, linewidth=2)

ax.set_title('AutoStemGNN', fontsize=22)
ax.set_ylabel('Values', fontsize=20)
ax.set_xlabel('Timestamp [D]', fontsize=20)
ax.legend(prop={'size': 15})
ax.grid()
plt.show()
gg = 1



