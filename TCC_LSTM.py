


from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.models import Sequential

data_atual = datetime.today().strftime('%Y-%m-%d')
data_inicio = '2021-01-01'
par_moeda = 'btc-usd'

#Capturar histórico de preços via biblioteca do Yahoo Finance
df_moeda = yf.download(par_moeda, data_inicio, data_atual)
df_moeda.reset_index(inplace=True)
#Selecionar campos do dataframe para análise
df_close_moeda = df_moeda[['Date', 'Adj Close']]
#Transformação do Index
df_close_moeda = df_close_moeda.set_index(pd.DatetimeIndex(df_close_moeda['Date'].values))
df_close_moeda.drop('Date', axis=1, inplace=True)
a = df_close_moeda
#a = df_moeda
print(a)
##########################   PLOT DAS INFORMAÇÕES   #########################################################
#plt.figure(figsize=(16,8))
#plt.title('Histórico ' + par_moeda)
#plt.plot(df_close_moeda['Adj Close'])
#plt.xlabel('Range de Datas')
#plt.ylabel('Preço de Fechamento')
#plt.show()

##########################   PLOT DAS INFORMAÇÕES   #########################################################

#Coletar total de linhas do dataframe para separar em treino e teste
total_linhas = len(df_close_moeda)
total_linhas_treino = round(.70 * total_linhas)
total_linhas_teste = total_linhas - total_linhas_treino
info = (
    f"linhas treino = 0:{total_linhas_treino}"
    f" linhas teste = {total_linhas_treino}:{total_linhas_treino + total_linhas_teste}")

print(info)

##########################   NORMALIZAR DADOS COLETADOS   #########################################################
scaler = StandardScaler()
df_moeda_normalizada = scaler.fit_transform(df_close_moeda)

##########################   SEPARAR DADOS PARA TREINO   #########################################################
train = df_moeda_normalizada[:total_linhas_treino]

##########################   SEPARAR DADOS PARA TESTE   #########################################################
test = df_moeda_normalizada[total_linhas_treino: total_linhas_treino + total_linhas_teste]

##########################  FORMATAR VALORES EM ARRAY PARA A REDE LSTM    #########################################################
def definir_lstm(df, step=1):
        dataX, dataY = [], []
        for i in range(len(df)-steps-1):
            a = df[i:(i+steps), 0]
            dataX.append(a)
            dataY.append(df[i + steps, 0])
        return np.array(dataX), np.array(dataY)

steps = 15
X_train, Y_train = definir_lstm(train, steps)
X_test, Y_test = definir_lstm(test, steps)

##########################  AJUSTAR OS DADOS PARA A REDE LSTM    #########################################################
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

##########################  ESTRUTURANDO A REDE    #########################################################
input_rede = Sequential()
# Definição de camadas na Rede
qtd_neuronios = 35
qtd_epocas = 100
input_rede.add(LSTM(qtd_neuronios, return_sequences=True, input_shape = (steps, 1)))
input_rede.add(LSTM(qtd_neuronios, return_sequences=True))
input_rede.add(LSTM(qtd_neuronios))
# Tratamento para evitar overfiting
input_rede.add(Dropout(0.2))
# Saída de 1 neurônio, onde irá retornar o valor de fechamento previsto
input_rede.add(Dense(1))
input_rede.compile(optimizer='adam', loss='mse')
input_rede.summary()

##########################  TREINAR MODELO CRIADO    #########################################################
treinamento = input_rede.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=qtd_epocas, batch_size=15, verbose=2)
##########################  Plotar Treinamento    #########################################################
plt.plot(treinamento.history['loss'], label='Training Loss')
plt.plot(treinamento.history['val_loss'], label='Validation Loss')
plt.legend()
plt.show()

##########################  Realizar previsão para dias seguintes    #########################################################
previsao = input_rede.predict(X_test)
#Inverter os dados normalizados para refletirem os valores reais, de acordo com as tendências
previsao = scaler.inverse_transform(previsao)
print(previsao)

tamanho_teste = len(test)
print ('tamanho_teste => ' + str(tamanho_teste))
dias_para_analise = tamanho_teste - steps
print ('dias_para_analise => ' + str(dias_para_analise))

##########################  TRANSFORMAR df_moeda_normalizada EM ARRAY   #########################################################
dados_teste = test[dias_para_analise:]
dados_teste = np.array(dados_teste).reshape(1, -1)
print('dados_teste => ' + str(dados_teste))

##########################  TRANSFORMAR EM LISTA   #########################################################
lista_saida_steps = list(dados_teste)
lista_saida_steps = lista_saida_steps[0].tolist()
print(lista_saida_steps)

list_prev_output = []
contador_dia = 0
qtd_dias_previsao = 10
while(contador_dia < qtd_dias_previsao):
    if(len(lista_saida_steps) > steps):
        dados_teste = np.array(lista_saida_steps[1:])
        #print("{} dia. Valores de entrada -> {}".format(contador_dia, dados_teste))
        dados_teste = dados_teste.reshape(1, -1)
        dados_teste = dados_teste.reshape((1, steps, -1))
        #print("dados_teste: ")
        #print(dados_teste)
        previsao = input_rede.predict(dados_teste, verbose = 0)
        print("{} dia. Valor previsto -> {}".format(contador_dia, previsao))
        lista_saida_steps.extend(previsao[0].tolist())
        lista_saida_steps = lista_saida_steps[1:]
        #print('lista_saida_steps: ')
        #print(lista_saida_steps)
        list_prev_output.extend(previsao.tolist())
        contador_dia += 1
    else:
        dados_teste = dados_teste.reshape((1, steps, -1))
        previsao = input_rede.predict(dados_teste, verbose=0)
        #print(previsao[0])
        lista_saida_steps.extend(previsao[0].tolist())
        #print(len(lista_saida_steps))
        list_prev_output.extend(previsao.tolist())
        contador_dia += 1

print(lista_saida_steps)
print('FIM!!!')

##########################  TRANSFORMAR A SAÍDA   #########################################################
previsao = scaler.inverse_transform(list_prev_output)
previsao = np.array(previsao).reshape(1, -1)
lista_saida_previsao = list(previsao)
lista_saida_previsao = previsao[0].tolist()
print(lista_saida_previsao)

##########################  TRATAR DATAS DE PREVISÃO   #########################################################
datas = pd.to_datetime(df_moeda['Date'])
previsao_datas = pd.date_range(list(datas)[-1] + pd.DateOffset(1), periods=10, freq='b').tolist()
print(previsao_datas)

datas_previstas = []
for i in previsao_datas:
    datas_previstas.append(i.date())

df_previsao = pd.DataFrame({'Data': np.array(datas_previstas), 'Preco_Previsto': lista_saida_previsao})
df_previsao['Data'] = pd.to_datetime(df_previsao['Data'])
df_previsao = df_previsao.set_index(pd.DatetimeIndex(df_previsao['Data'].values))
df_previsao.drop('Data', axis=1, inplace=True)
print(df_previsao)

plt.figure(figsize=(16, 8))
plt.title('Histórico ' + par_moeda)
plt.plot(df_close_moeda['Adj Close'])
plt.plot(df_previsao['Preco_Previsto'])
plt.xlabel('Range de Datas')
plt.ylabel('Preço de Fechamento')
plt.legend(['Preco de Fechamento', 'Preco previsto'])
plt.show()



