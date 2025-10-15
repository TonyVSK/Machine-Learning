import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_curve, auc, RocCurveDisplay
import warnings
warnings.filterwarnings('ignore')
sns.set_style("darkgrid")


# Os dados serão tratados de maneira identica ao colab da aula que trata de um churn também, usamos portanto o random florest
df = pd.read_csv('telecom_churn.csv')



df.drop('phone number', axis=1, inplace=True)
df['international plan'] = df['international plan'].map({'yes': 1, 'no': 0})
df['voice mail plan'] = df['voice mail plan'].map({'yes': 1, 'no': 0})
df['churn'] = df['churn'].map({True: 1, False: 0})
df = pd.get_dummies(df, columns=['state', 'area code'], drop_first=True)


X = df.drop('churn', axis=1) # aqui separamos as features e target
y = df['churn']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)


numerical_cols = ['account length', 'number vmail messages', 'total day minutes', 'total day calls',
                  'total day charge', 'total eve minutes', 'total eve calls', 'total eve charge',
                  'total night minutes', 'total night calls', 'total night charge',
                  'total intl minutes', 'total intl calls', 'total intl charge',
                  'customer service calls']

scaler = StandardScaler()
X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X_train, y_train)


rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_leaf=5, random_state=42)
rf_model.fit(X_res, y_res)


y_pred = rf_model.predict(X_test)




# criação de matriz de Confusão
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Churn (0)', 'Churn (1)'])
# Aqui eu adicionei uma função para salvar as imagens, não sabia fazer isso, então pedi pro chatgpt gerar para mim 
## Todas as declarações para salvar imagens foram feitas com auxilio do chatgpt (depois vi que era só adicionar plt.savefig e o nome que eu gostaria)

# Criar a figura
fig, ax = plt.subplots(figsize=(6, 6))
disp.plot(cmap=plt.cm.Blues, ax=ax)
plt.title('Matriz de Confusão - Random Forest')
plt.tight_layout() # Ajusta o layout para evitar cortes
plt.savefig('matriz_de_confusao_rf.png') # SALVA A PRIMEIRA FIGURA
plt.close(fig) # Fecha a figura para liberar memória


feature_importances = pd.Series(rf_model.feature_importances_, index=X_train.columns).sort_values(ascending=False).head(10)

# Criar a figura
fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x=feature_importances.values, y=feature_importances.index, ax=ax)
plt.title('Features mais importantes')
plt.xlabel('Importância')
plt.ylabel('Feature')
plt.tight_layout()
plt.savefig('importancia_de_features_rf.png') # SALVA A SEGUNDA FIGURA
plt.close(fig) # Fecha a figura

print("\nAs figuras foram salvas como 'matriz_de_confusao_rf.png' e 'importancia_de_features_rf.png'.")



warnings.filterwarnings('ignore')
sns.set_style("darkgrid")
df = pd.read_csv('telecom_churn.csv')
# Converti o true ou false para sim ou não para ficar mais intuitivo
df['churn_label'] = df['churn'].map({True: 'Sim', False: 'Não'})


# Além do gráfico com a matriz de confusão, eu gerei também gráficos semelhantes ao que foi exibido no colab para podermos analisar, o png das imagens 





#Distribuição da Variável Alvo (Churn)
plt.figure(figsize=(6, 5))
sns.countplot(x='churn_label', data=df)
plt.title('Distribuição de Clientes: Churn vs. Não Churn')
plt.xlabel('Churn')
plt.ylabel('Número de Clientes')
plt.text(0, df['churn_label'].value_counts()['Não'] + 50, f"Não Churn: {df['churn_label'].value_counts()['Não']}", ha='center')
plt.text(1, df['churn_label'].value_counts()['Sim'] + 50, f"Churn: {df['churn_label'].value_counts()['Sim']}", ha='center')
plt.savefig('eda_churn_distribuicao.png')
plt.show()
#Relação entre o Custo Diário e o Churn (Boxplot)
plt.figure(figsize=(8, 6))
sns.boxplot(x='churn_label', y='total day charge', data=df)
plt.title('Boxplot do Custo Diário Total por Churn')
plt.xlabel('Churn')
plt.ylabel('Custo Diário Total')
plt.savefig('eda_custo_diario_boxplot.png')
plt.show()
#Relaçap entre Plano Internacional e Churn (Taxa)
plt.figure(figsize=(7, 6))
churn_rate_plan = df.groupby('international plan')['churn'].mean().reset_index()
sns.barplot(x='international plan', y='churn', data=churn_rate_plan, palette='viridis')
plt.title('Taxa de Churn por Plano Internacional')
plt.xlabel('Plano Internacional')
plt.ylabel('Taxa de Churn (Média de Churn)')
plt.ylim(0, churn_rate_plan['churn'].max() * 1.2)
plt.savefig('eda_churn_por_plano_int.png')
plt.show()
#Relação entre Chamadas de Atendimento ao Cliente e Churn
plt.figure(figsize=(10, 6))
sns.countplot(x='customer service calls', hue='churn_label', data=df)
plt.title('Contagem de Churn por Número de Chamadas de Serviço')
plt.xlabel('Número de Chamadas de Serviço')
plt.ylabel('Contagem de Clientes')
plt.legend(title='Churn')
plt.savefig('eda_churn_por_chamadas_servico.png')
plt.show()



# O gráfico de distribuição com minutos diários por Churn que eu tentei gerar não ficou parecido com as amostras feitas no colab, porque as colunas e os dados eram diferentes já que haviam
# muitas váriaveis de charges, o que me confundiu sobre o que tratar, poratnto gerei um gráfico de distribuição por curvas
plt.figure(figsize=(10, 6))
sns.kdeplot(data=df, x='total day minutes', hue='churn_label', fill=True, alpha=.5, linewidth=2)
plt.title('Distribuição de Minutos Diários por Churn')
plt.xlabel('Minutos Diários Totais')
plt.ylabel('Densidade')
plt.legend(title='Churn')
plt.savefig('eda_distribuicao_minutos_churn.png')
plt.show()





# Figura Distribuição de Clientes: Churn vs. Não Churn

# O resultado obtido foi totalmente desbablanceado, já que tem muito mais clientes que cancelaram o serviço do que aceitaram



# Figura de boxplot do Custo Diário Total por Churn

# Os clientes que deram churn = Sim tendem a ter custos diários médios mais altos, com uma dispersão maior nos valores. Portanto, é possivelmente
# imaginar que quanto maior gasto diário, mais propensos a cancelar estão, podendo ser por insatisfação em relação aos gastos.




# Figura da taxa de Churn por plano internacional

# É muito maior entre clientes com plano internacional (≈ 43%) do que entre clientes sem o plano, portanto o plano internacional É
# um valor extremamente relevante para o cancelamento




# Figura da contagem de Churn por número de chamadas de serviço

# Quanto mais chamadas de atendimento o cliente faz, maior a chance de churn. Ou seja, clientes que contatam o suporte várias vezes
# são mais propensos a cancelar, provavelmente por insatisfação






# Figura da distribuição de min diários por Churn

# Os clientes que cancelaram apresentam padrões de uso mais variados, onde temos alguns deles com mais minutos em média.
# conclui-se que clientes com mais uso podem achar que não vale a pena o gasto com o serviço oferecido




# Figura da Matriz de Confusão com o modelo Random Forest

# O modelo Random Forest obteve bom desempenho,  a maioria dos casos de “Não Churn” foram acertadas, com 807 acertos, e uma quantidade aceitável de “Churn” também, com 97 acertos.
# Evidentemente, os resultados não foram perfeitos, já que tivemos 48 falsos negativos:clientes que cancelaram, mas o modelo não conseguiu prever.
# Esses clientessão os mais críticos para a empresa, pois representam perdas imprevistas.