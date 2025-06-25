import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Carregar os dados
@st.cache_data
def load_data():
    df = pd.read_csv("BTC_preds_consolidadas.csv", parse_dates=["Date"])
    return df

df = load_data()

# Título
st.title("Previsões da Bitcoin - Modelos de IA")

# Seleção de modelos
modelos = [col for col in df.columns if col not in ["Date", "Real"]]
modelo_escolhido = st.selectbox("Seleciona o modelo para visualizar:", modelos)

# Mostrar gráfico
st.subheader(f"Comparação entre o valor real e a previsão do modelo {modelo_escolhido}")
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(df["Date"], df["Real"], label="Real", marker="o")
ax.plot(df["Date"], df[modelo_escolhido], label=modelo_escolhido, marker="x")
ax.set_xlabel("Data")
ax.set_ylabel("Valor da BTC")
ax.legend()
ax.grid(True)
st.pyplot(fig)

# Cálculo de métricas de erro
st.subheader("Métricas de Avaliação")
y_true = df["Real"]
y_pred = df[modelo_escolhido]
mae = abs(y_true - y_pred).mean()
rmse = ((y_true - y_pred) ** 2).mean() ** 0.5
mape = (abs((y_true - y_pred) / y_true)).mean()

st.write(f"**MAE:** {mae:.2f}")
st.write(f"**RMSE:** {rmse:.2f}")
st.write(f"**MAPE:** {mape:.4f}")
