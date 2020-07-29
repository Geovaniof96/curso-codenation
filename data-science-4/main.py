#!/usr/bin/env python
# coding: utf-8

# # Desafio 6
# 
# Neste desafio, vamos praticar _feature engineering_, um dos processos mais importantes e trabalhosos de ML. Utilizaremos o _data set_ [Countries of the world](https://www.kaggle.com/fernandol/countries-of-the-world), que contém dados sobre os 227 países do mundo com informações sobre tamanho da população, área, imigração e setores de produção.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Setup_ geral

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import sklearn as sk
from sklearn.preprocessing import (
    OneHotEncoder, Binarizer, KBinsDiscretizer,
    MinMaxScaler, StandardScaler, PolynomialFeatures)
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


# In[2]:


# Algumas configurações para o matplotlib.
#%matplotlib inline

#from IPython.core.pylabtools import figsize


#figsize(12, 8)

#sns.set()


# In[3]:


countries = pd.read_csv("countries.csv")


# In[4]:


new_column_names = [
    "Country", "Region", "Population", "Area", "Pop_density", "Coastline_ratio",
    "Net_migration", "Infant_mortality", "GDP", "Literacy", "Phones_per_1000",
    "Arable", "Crops", "Other", "Climate", "Birthrate", "Deathrate", "Agriculture",
    "Industry", "Service"
]

countries.columns = new_column_names

countries.head(5)


# ## Observações
# 
# Esse _data set_ ainda precisa de alguns ajustes iniciais. Primeiro, note que as variáveis numéricas estão usando vírgula como separador decimal e estão codificadas como strings. Corrija isso antes de continuar: transforme essas variáveis em numéricas adequadamente.
# 
# Além disso, as variáveis `Country` e `Region` possuem espaços a mais no começo e no final da string. Você pode utilizar o método `str.strip()` para remover esses espaços.

# ## Inicia sua análise a partir daqui

# #### Organizando 

# In[5]:


countries.dtypes


# In[6]:


#Fazendo a troca das "," por "."
countries.iloc[:,[4,5,6,7,9,10,11,12,13,14,15,16,17,18,19]] = countries.iloc[:,[4,5,6,7,9,10,11,12,13,14,15,16,17,18,19]]                                                                    .apply(lambda x: x.str.replace(',' , '.'))

# Convertendo os elementos que são float para o tipo object
countries.iloc[:,[4,5,6,7,9,10,11,12,13,14,15,16,17,18,19]] = countries.iloc[:,[4,5,6,7,9,10,11,12,13,14,15,16,17,18,19]]                                                                    .astype(float)


# In[7]:


countries.dtypes


# In[8]:


# Removendo espaços desnecessários das colunas Country e Region com a função str.strip
countries.loc[:,['Country', 'Region']] = countries.loc[:,['Country', 'Region']].applymap(str.strip)


# ## Questão 1
# 
# Quais são as regiões (variável `Region`) presentes no _data set_? Retorne uma lista com as regiões únicas do _data set_ com os espaços à frente e atrás da string removidos (mas mantenha pontuação: ponto, hífen etc) e ordenadas em ordem alfabética.

# In[9]:


def q1():
    
    # Gerando a lista de regiões únicas
    lista = list(countries.Region.unique())
    
    # Ordenando a lista
    lista.sort()
    
    # Visualizando a lista
    return lista


# In[10]:


q1()


# ## Questão 2
# 
# Discretizando a variável `Pop_density` em 10 intervalos com `KBinsDiscretizer`, seguindo o encode `ordinal` e estratégia `quantile`, quantos países se encontram acima do 90º percentil? Responda como um único escalar inteiro.

# In[139]:


def q2():
    #Aplicando a discretização
         #Formato requetido
    discretizer = KBinsDiscretizer(n_bins=10, encode="ordinal", strategy="quantile")
       
        #Ajustando o sistema para que aprenda esses valores
    discretizer.fit(countries[["Pop_density"]])

        #Fazendo as transformações para cada um dos intervalos e plotando
    ajuste = discretizer.transform(countries[["Pop_density"]])
    
    # Filtrando e contando valores maiores que o quartil
    qnt = int((ajuste > np.quantile(ajuste, 0.9)).sum())

    return qnt


# In[140]:


q2()


# # Questão 3
# 
# Se codificarmos as variáveis `Region` e `Climate` usando _one-hot encoding_, quantos novos atributos seriam criados? Responda como um único escalar.

# In[164]:


def q3():
    
    # Removendo Valores NAN
    countries1 = countries.dropna()
        
    # Configurando o encoder
    encoder = OneHotEncoder(sparse=False, dtype=np.int, )
    
    # Aplicando o encoder nas colunas Region e Climate
    data_encoded = encoder.fit_transform(countries1[['Region','Climate']])
    
    return (data_encoded.shape[1]+1)


# In[165]:


q3()


# ## Questão 4
# 
# Aplique o seguinte _pipeline_:
# 
# 1. Preencha as variáveis do tipo `int64` e `float64` com suas respectivas medianas.
# 2. Padronize essas variáveis.
# 
# Após aplicado o _pipeline_ descrito acima aos dados (somente nas variáveis dos tipos especificados), aplique o mesmo _pipeline_ (ou `ColumnTransformer`) ao dado abaixo. Qual o valor da variável `Arable` após o _pipeline_? Responda como um único float arredondado para três casas decimais.

# In[43]:


test_country = [
    'Test Country', 'NEAR EAST', -0.19032480757326514,
    -0.3232636124824411, -0.04421734470810142, -0.27528113360605316,
    0.13255850810281325, -0.8054845935643491, 1.0119784924248225,
    0.6189182532646624, 1.0074863283776458, 0.20239896852403538,
    -0.043678728558593366, -0.13929748680369286, 1.3163604645710438,
    -0.3699637766938669, -0.6149300604558857, -0.854369594993175,
    0.263445277972641, 0.5712416961268142
]


# In[168]:


def q4():
   
    #Aplicando a sequencia
    num_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())])
    
    # Treinando o pipeline para as colunas que possuem numemos(np.number)
    num_pipeline.fit(countries.select_dtypes([np.number]))
    
    #Obtendo os valores das medianas das colunas em questão
    medianas = num_pipeline.transform([test_country[2:]])
    
    return float(medianas[0][9].round(3))


# In[169]:


q4()


# ## Questão 5
# 
# Descubra o número de _outliers_ da variável `Net_migration` segundo o método do _boxplot_, ou seja, usando a lógica:
# 
# $$x \notin [Q1 - 1.5 \times \text{IQR}, Q3 + 1.5 \times \text{IQR}] \Rightarrow x \text{ é outlier}$$
# 
# que se encontram no grupo inferior e no grupo superior.
# 
# Você deveria remover da análise as observações consideradas _outliers_ segundo esse método? Responda como uma tupla de três elementos `(outliers_abaixo, outliers_acima, removeria?)` ((int, int, bool)).

# In[141]:


def q5():
    
    aux = countries['Net_migration']
    
    q1 = aux.quantile(0.25)
    q3 = aux.quantile(0.75)
    iqr = q3 - q1

    faixa = [q1 - 1.5 * iqr, q3 + 1.5 * iqr]
    
    abaixo = int((aux < faixa[0]).sum())
    acima = int((aux > faixa[1]).sum())
    
    qnt_fora = abaixo + acima
    total = aux.shape[0]
    
    remover = bool( qnt_fora/total < 0.05 )
    
    return (abaixo,acima,remover)
    


# In[142]:


q5()


# ## Questão 6
# Para as questões 6 e 7 utilize a biblioteca `fetch_20newsgroups` de datasets de test do `sklearn`
# 
# Considere carregar as seguintes categorias e o dataset `newsgroups`:
# 
# ```
# categories = ['sci.electronics', 'comp.graphics', 'rec.motorcycles']
# newsgroup = fetch_20newsgroups(subset="train", categories=categories, shuffle=True, random_state=42)
# ```
# 
# 
# Aplique `CountVectorizer` ao _data set_ `newsgroups` e descubra o número de vezes que a palavra _phone_ aparece no corpus. Responda como um único escalar.

# In[128]:


from sklearn.datasets import fetch_20newsgroups 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


# In[145]:


#Aplicando as categorias
categories = ['sci.electronics', 'comp.graphics', 'rec.motorcycles']
newsgroup = fetch_20newsgroups(subset="train", categories=categories, shuffle=True, random_state=42)
    


# In[166]:


def q6():
    
    #Aplicando as condiçoes para a contagem 
    count_vectorizer = CountVectorizer()
    newsgroups_counts = count_vectorizer.fit_transform(newsgroup.data)
    
    return int(newsgroups_counts[:,count_vectorizer.vocabulary_['phone']].sum())
    


# In[167]:


q6()


# ## Questão 7
# 
# Aplique `TfidfVectorizer` ao _data set_ `newsgroups` e descubra o TF-IDF da palavra _phone_. Responda como um único escalar arredondado para três casas decimais.

# In[162]:


def q7():
    
    #Aplicando o TfidfVectorizer
    tfidf_vectorizer = TfidfVectorizer()

    #Ajustando os valores
    words = tfidf_vectorizer.fit_transform(newsgroup.data)


    return round(float(words[:, tfidf_vectorizer.vocabulary_['phone']].sum()),3)


# In[163]:


q7()


# In[ ]:




