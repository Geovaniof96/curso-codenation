#!/usr/bin/env python
# coding: utf-8

# # Desafio 3
# 
# Neste desafio, iremos praticar nossos conhecimentos sobre distribuições de probabilidade. Para isso,
# dividiremos este desafio em duas partes:
#     
# 1. A primeira parte contará com 3 questões sobre um *data set* artificial com dados de uma amostra normal e
#     uma binomial.
# 2. A segunda parte será sobre a análise da distribuição de uma variável do _data set_ [Pulsar Star](https://archive.ics.uci.edu/ml/datasets/HTRU2), contendo 2 questões.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Setup_ geral

# In[63]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sct
import seaborn as sns
from statsmodels.distributions.empirical_distribution import ECDF


# In[64]:


#%matplotlib inline

#from IPython.core.pylabtools import figsize

#figsize(12, 8)
#sns.set()


# In[ ]:





# ## Parte 1

# ### _Setup_ da parte 1

# In[65]:



np.random.seed(42)
    
dataframe = pd.DataFrame({"normal": sct.norm.rvs(20, 4, size=10000),
                     "binomial": sct.binom.rvs(100, 0.2, size=10000)})


# In[66]:


sns.distplot(dataframe['normal']);


# In[67]:


sns.distplot(dataframe['binomial']);


# ## Inicie sua análise a partir da parte 1 a partir daqui

# In[68]:


# Sua análise da parte 1 começa aqui.


# In[69]:


dataframe.head(5)


# In[70]:


#Pegando os valores das médias e dos desvios

media=dataframe.mean()
dv=dataframe.std()

print("A média é:")
print(media)
print("\n")
print("O desvio padrão é:")
print(dv)


# In[71]:


#Separando os quartis

dataframe.quantile([0.25,0.5,0.75])


# ## Questão 1
# 
# Qual a diferença entre os quartis (Q1, Q2 e Q3) das variáveis `normal` e `binomial` de `dataframe`? Responda como uma tupla de três elementos arredondados para três casas decimais.
# 
# Em outra palavras, sejam `q1_norm`, `q2_norm` e `q3_norm` os quantis da variável `normal` e `q1_binom`, `q2_binom` e `q3_binom` os quantis da variável `binom`, qual a diferença `(q1_norm - q1 binom, q2_norm - q2_binom, q3_norm - q3_binom)`?

# In[72]:


#Separando cada quartis

n1,n2,n3 = dataframe['normal'].quantile([0.25,0.50,0.75])
b1,b2,b3 = dataframe['binomial'].quantile([0.25,0.50,0.75])


# In[73]:


def q1():
    
    #Observando as diferenças
    Q1 = round(n1-b1,3)
    Q2 = round(n2-b2,3)
    Q3 = round(n3-b3,3)

    return (Q1,Q2,Q3)


# In[118]:


#Resposta
q1()


# Para refletir:
# 
# * Você esperava valores dessa magnitude?
# 
# 
# 
# * Você é capaz de explicar como distribuições aparentemente tão diferentes (discreta e contínua, por exemplo) conseguem dar esses valores?

# ## Questão 2
# 
# Considere o intervalo $[\bar{x} - s, \bar{x} + s]$, onde $\bar{x}$ é a média amostral e $s$ é o desvio padrão. Qual a probabilidade nesse intervalo, calculada pela função de distribuição acumulada empírica (CDF empírica) da variável `normal`? Responda como uma único escalar arredondado para três casas decimais.

# Para pegar a porcentagem do intervalo se faz necessário achar a probabilidade acumulada em cada um dos 
# pontos e após fazer a diferença
# 

# In[74]:


def q2():
    
    #Atribuindo o dataframe para um CDF empirico
    ecdf = ECDF(dataframe['normal'])

    #Achando os valores das pobabilidades de cada ponta do intervalo 
    down = ecdf(media['normal']-dv['normal'])
    up = ecdf(media['normal'] + dv['normal'])
    
    return (round(float(up-down),3))


# In[75]:


q2()


# Para refletir:
# 
# * Esse valor se aproxima do esperado teórico?
# * Experimente também para os intervalos $[\bar{x} - 2s, \bar{x} + 2s]$ e $[\bar{x} - 3s, \bar{x} + 3s]$.

# ## Questão 3
# 
# Qual é a diferença entre as médias e as variâncias das variáveis `binomial` e `normal`? Responda como uma tupla de dois elementos arredondados para três casas decimais.
# 
# Em outras palavras, sejam `m_binom` e `v_binom` a média e a variância da variável `binomial`, e `m_norm` e `v_norm` a média e a variância da variável `normal`. Quais as diferenças `(m_binom - m_norm, v_binom - v_norm)`?

# In[119]:


def q3():   
    #Obtendo os valores das médias e das variâncias - a media ja tinha sido obtido anteriormente 
    
    var = dataframe.var()
    
    #Obtendo as difenrenças
    
    M = media['binomial']-media['normal']
    V = var['binomial']-var['normal']
    
    return (tuple(np.round((M,V),3)))


# Para refletir:
# 
# * Você esperava valore dessa magnitude?
# * Qual o efeito de aumentar ou diminuir $n$ (atualmente 100) na distribuição da variável `binomial`?

# ## Parte 2

# ### _Setup_ da parte 2

# In[77]:


stars = pd.read_csv("pulsar_stars.csv")

stars.rename({old_name: new_name
              for (old_name, new_name)
              in zip(stars.columns,
                     ["mean_profile", "sd_profile", "kurt_profile", "skew_profile", "mean_curve", "sd_curve", "kurt_curve", "skew_curve", "target"])
             },
             axis=1, inplace=True)

stars.loc[:, "target"] = stars.target.astype(bool)


# ## Inicie sua análise da parte 2 a partir daqui

# In[78]:


# Sua análise da parte 2 começa aqui.


# In[79]:


stars.head(5)


# ## Questão 4
# 
# Considerando a variável `mean_profile` de `stars`:
# 
# 1. Filtre apenas os valores de `mean_profile` onde `target == 0` (ou seja, onde a estrela não é um pulsar).
# 2. Padronize a variável `mean_profile` filtrada anteriormente para ter média 0 e variância 1.
# 
# Chamaremos a variável resultante de `false_pulsar_mean_profile_standardized`.
# 
# Encontre os quantis teóricos para uma distribuição normal de média 0 e variância 1 para 0.80, 0.90 e 0.95 através da função `norm.ppf()` disponível em `scipy.stats`.
# 
# Quais as probabilidade associadas a esses quantis utilizando a CDF empírica da variável `false_pulsar_mean_profile_standardized`? Responda como uma tupla de três elementos arredondados para três casas decimais.

# In[80]:


# Filtrar valores para questão 4 e 5
valores_filtrados = stars.loc[stars['target'] == False ,'mean_profile']
    
# Padronizar
valores_filtrados = (valores_filtrados - valores_filtrados.mean())/valores_filtrados.std()


# In[117]:


def q4():
    # Calcular os quantiles de uma distribuição normal
    Q80, Q90, Q95 = sct.norm.ppf([0.80, 0.90, 0.95], loc = 0, scale = 1)
    
    #Retornar um CDF empirico dos dados filtrados
    ecdf = ECDF(valores_filtrados)
    
    Q4=tuple(np.round(ecdf([Q80, Q90, Q95]), 3))
    
    return Q4


# In[104]:


q4()


# Para refletir:
# 
# * Os valores encontrados fazem sentido?
# * O que isso pode dizer sobre a distribuição da variável `false_pulsar_mean_profile_standardized`?

# ## Questão 5
# 
# Qual a diferença entre os quantis Q1, Q2 e Q3 de `false_pulsar_mean_profile_standardized` e os mesmos quantis teóricos de uma distribuição normal de média 0 e variância 1? Responda como uma tupla de três elementos arredondados para três casas decimais.

# In[116]:


def q5():
    
    #Obtindo as quantis dos valores filtrados
    Q_25, Q_50, Q_75 = valores_filtrados.quantile([0.25, 0.5, 0.75])
    
    #Obtendo os quantis com as caracteristicas requeridas
    Q_25n, Q_50n, Q_75n = sct.norm.ppf([0.25, 0.5, 0.75], loc = 0, scale = 1)
    
    return (tuple(np.round((Q_25-Q_25n,Q_50-Q_50n,Q_75-Q_75n),3)))


# Para refletir:
# 
# * Os valores encontrados fazem sentido?
# * O que isso pode dizer sobre a distribuição da variável `false_pulsar_mean_profile_standardized`?
# * Curiosidade: alguns testes de hipóteses sobre normalidade dos dados utilizam essa mesma abordagem.
