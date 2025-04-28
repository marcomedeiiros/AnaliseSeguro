import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

df = pd.read_csv(r'C:\Users\maravilhoso\Downloads\seguro_carro_dataset.csv')

plt.figure(figsize=(19.20, 10.80))
sns.barplot(x=df['Idade'].value_counts().index, y=df['Idade'].value_counts().values, color='skyblue')
plt.title('Distribuição da Idade', fontsize=22)
plt.xlabel('Idade', fontsize=18)
plt.ylabel('Contagem', fontsize=18)

for p in plt.gca().patches:
    plt.gca().annotate(f'{p.get_height()}',
                       (p.get_x() + p.get_width() / 2., p.get_height()),
                       xytext=(0, 10),
                       textcoords='offset points',
                       ha='center', va='bottom', fontsize=14, color='black')

mng = plt.get_current_fig_manager()
mng.window.state('zoomed')

plt.tight_layout()
plt.savefig('grafico_idade.png', dpi=300)  
plt.show()

bins = [0, 50000, 100000, 150000, 200000, 250000, 300000, np.inf]
labels = ['0-50k', '50k-100k', '100k-150k', '150k-200k', '200k-250k', '250k-300k', '300k+']
df['Faixa Valor Carro'] = pd.cut(df['Valor do Carro'], bins=bins, labels=labels)

plt.figure(figsize=(19.20, 10.80))
sns.countplot(x='Faixa Valor Carro', data=df, color='lightcoral')
plt.title('Distribuição do Valor do Carro por Faixas', fontsize=22)
plt.xlabel('Faixa de Valor do Carro', fontsize=18)
plt.ylabel('Contagem', fontsize=18)

for p in plt.gca().patches:
    plt.gca().annotate(f'{p.get_height()}',
                       (p.get_x() + p.get_width() / 2., p.get_height()),
                       xytext=(0, 10),
                       textcoords='offset points',
                       ha='center', va='bottom', fontsize=14, color='black')

mng = plt.get_current_fig_manager()
mng.window.state('zoomed')

plt.tight_layout()
plt.savefig('grafico_valor_carro.png', dpi=300)
plt.show()

X = df[['Idade', 'Tempo de Habilitação (anos)', 'Valor do Carro', 'Histórico de Sinistro', 'Classe de Bônus']]
y = df['Valor do Seguro']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

modelo = LinearRegression()
modelo.fit(X_train, y_train)

y_pred = modelo.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(f'Mean Squared Error: {mse}')
print(f'Root Mean Squared Error: {rmse}')

plt.figure(figsize=(19.20, 10.80))

plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred, alpha=0.6, color='blue')
plt.xlabel('Valor Real do Seguro', fontsize=16)
plt.ylabel('Valor Previsto do Seguro', fontsize=16)
for i in range(len(y_test)):
    plt.text(y_test.iloc[i], y_pred[i], f'{y_test.iloc[i]:.2f}', fontsize=8, alpha=0.7)

plt.subplot(1, 2, 2)
plt.scatter(y_pred, y_test, alpha=0.6, color='green')
plt.xlabel('Valor Previsto do Seguro', fontsize=16)
plt.ylabel('Valor Real do Seguro', fontsize=16)
for i in range(len(y_test)):
    plt.text(y_pred[i], y_test.iloc[i], f'{y_pred[i]:.2f}', fontsize=8, alpha=0.7)

mng = plt.get_current_fig_manager()
mng.window.state('zoomed')

plt.tight_layout()
plt.savefig('grafico_regressao.png', dpi=300)
plt.show()