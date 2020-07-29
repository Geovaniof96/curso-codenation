#!/usr/bin/env python
# coding: utf-8

# # Desafio 5
# 
# Neste desafio, vamos praticar sobre redução de dimensionalidade com PCA e seleção de variáveis com RFE. Utilizaremos o _data set_ [Fifa 2019](https://www.kaggle.com/karangadiya/fifa19), contendo originalmente 89 variáveis de mais de 18 mil jogadores do _game_ FIFA 2019.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Setup_ geral

# In[4]:


from math import sqrt

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sct
import seaborn as sns
import statsmodels.api as sm
import statsmodels.stats as st
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression

from loguru import logger


# In[6]:


fifa = pd.read_csv("fifa.csv")


# In[7]:


columns_to_drop = ["Unnamed: 0", "ID", "Name", "Photo", "Nationality", "Flag",
                   "Club", "Club Logo", "Value", "Wage", "Special", "Preferred Foot",
                   "International Reputation", "Weak Foot", "Skill Moves", "Work Rate",
                   "Body Type", "Real Face", "Position", "Jersey Number", "Joined",
                   "Loaned From", "Contract Valid Until", "Height", "Weight", "LS",
                   "ST", "RS", "LW", "LF", "CF", "RF", "RW", "LAM", "CAM", "RAM", "LM",
                   "LCM", "CM", "RCM", "RM", "LWB", "LDM", "CDM", "RDM", "RWB", "LB", "LCB",
                   "CB", "RCB", "RB", "Release Clause"
]

try:
    fifa.drop(columns_to_drop, axis=1, inplace=True)
except KeyError:
    logger.warning(f"Columns already dropped")


# ## Inicia sua análise a partir daqui

# In[8]:


# Sua análise começa aqui.

#Conhecendo os dados que estamos trabalhando
fifa.head(5)


# In[9]:


dados = pd.DataFrame({'colunas': fifa.columns,
                    'tipo': fifa.dtypes,
                    'nulos': fifa.isna().sum(),
                    'size': fifa.shape[0],
                    'unicos': fifa.nunique()})
dados['percentual'] = round(dados['nulos'] / dados['size'],2)


# In[10]:


dados[dados.percentual == 0]['tipo'].value_counts()


# In[11]:


dados.percentual.plot.hist( bins = 5)


# Observando o gráfico acima, todos os valores percentuais são 0.0, logo todos são completos

# In[12]:


#Para realizar que o sistema aprenda, o que vamos pedir se faz necessário retirar todos os valores que não existem
fifa.dropna(inplace=True)


# In[13]:


#plotando gráfico para saber como os pontos estão distribuidos 
pca = PCA()
projected = pca.fit_transform(fifa)
print(f"Original shape: {fifa.shape}, projected shape: {projected.shape}")


# In[14]:


sns.scatterplot(projected[:, 0], projected[:, 1]);


# ### Gráfico para a questão 2

# In[15]:


evr = pca.explained_variance_ratio_

g = sns.lineplot(np.arange(len(evr)), np.cumsum(evr))
g.axes.axhline(0.95, ls="--", color="red")
plt.xlabel('Numeros de componentes')
plt.ylabel('Valor acumulado da variancia');


# ## Questão 1
# 
# Qual fração da variância consegue ser explicada pelo primeiro componente principal de `fifa`? Responda como um único float (entre 0 e 1) arredondado para três casas decimais.

# In[16]:


def q1():
    
    #Atribuindo o PCA e ensinando o sistema qual o modelo que temos
    pca = PCA()
    pca.fit(fifa) #fazendo o sistema aprender o modelo que temos
    
    p_variavel = pca.explained_variance_ratio_[0]
    
    return (float(round(p_variavel,3)))


# In[17]:


q1()


# ## Questão 2
# 
# Quantos componentes principais precisamos para explicar 95% da variância total? Responda como un único escalar inteiro.

# In[18]:


def q2():
    
    #Para uma variancia de 95%
    pca_095 = PCA(0.95)
    
    #Pegando o ponto onde as linhas se interceptam 
    X_reduced = pca_095.fit_transform(fifa)

    return X_reduced.shape[1]


# In[19]:


q2()


# ## Questão 3
# 
# Qual são as coordenadas (primeiro e segundo componentes principais) do ponto `x` abaixo? O vetor abaixo já está centralizado. Cuidado para __não__ centralizar o vetor novamente (por exemplo, invocando `PCA.transform()` nele). Responda como uma tupla de float arredondados para três casas decimais.

# In[20]:


x = [0.87747123,  -1.24990363,  -1.3191255, -36.7341814,
     -35.55091139, -37.29814417, -28.68671182, -30.90902583,
     -42.37100061, -32.17082438, -28.86315326, -22.71193348,
     -38.36945867, -20.61407566, -22.72696734, -25.50360703,
     2.16339005, -27.96657305, -33.46004736,  -5.08943224,
     -30.21994603,   3.68803348, -36.10997302, -30.86899058,
     -22.69827634, -37.95847789, -22.40090313, -30.54859849,
     -26.64827358, -19.28162344, -34.69783578, -34.6614351,
     48.38377664,  47.60840355,  45.76793876,  44.61110193,
     49.28911284
]


# In[21]:


def q3():

    pca = PCA(n_components=2)
    q3 = pca.fit(fifa)

    return(tuple(q3.components_.dot(x).round(3)))


# In[22]:


q3()


# ## Questão 4
# 
# Realiza RFE com estimador de regressão linear para selecionar cinco variáveis, eliminando uma a uma. Quais são as variáveis selecionadas? Responda como uma lista de nomes de variáveis.

# In[23]:


def q4():
    
    #Filtrando os dados
    x = fifa.drop('Overall', axis=1)
    y = fifa.Overall
    
    #Modelo utilizado
    modelo = LinearRegression()
    
    #Ajustando modelo
    rfe = RFE(modelo, 5)
    fit = rfe.fit(x, y)
    return list(x.loc[:,fit.support_].columns)


# In[24]:


q4()

