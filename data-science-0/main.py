#!/usr/bin/env python
# coding: utf-8

# # Desafio 1
# 
# Para esse desafio, vamos trabalhar com o data set [Black Friday](https://www.kaggle.com/mehdidag/black-friday), que reúne dados sobre transações de compras em uma loja de varejo.
# 
# Vamos utilizá-lo para praticar a exploração de data sets utilizando pandas. Você pode fazer toda análise neste mesmo notebook, mas as resposta devem estar nos locais indicados.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Set up_ da análise

# In[2]:


import pandas as pd
import numpy as np


# In[3]:


black_friday = pd.read_csv("black_friday.csv")


# ## Inicie sua análise a partir daqui

# In[4]:


black_friday.head(2)


# In[5]:


black_friday.columns


# ## Questão 1
# 
# Quantas observações e quantas colunas há no dataset? Responda no formato de uma tuple `(n_observacoes, n_colunas)`.

# In[6]:


black_friday.shape


# In[7]:


def q1():
    # Retorne aqui o resultado da questão 1.
    return black_friday.shape
    pass


# ## Questão 2
# 
# Há quantas mulheres com idade entre 26 e 35 anos no dataset? Responda como um único escalar.

# In[8]:


black_friday[(black_friday.Age == '26-35') & (black_friday.Gender == 'F')].shape[0]


# In[9]:


Q2=black_friday[(black_friday.Age == '26-35') & (black_friday.Gender == 'F')].shape[0]


# In[10]:


def q2():
    # Retorne aqui o resultado da questão 2.
    return Q2
    pass


# ## Questão 3
# 
# Quantos usuários únicos há no dataset? Responda como um único escalar.

# In[11]:


black_friday['User_ID'].unique()


# In[12]:


black_friday['User_ID'].unique().shape[0]


# In[13]:


def q3():
    # Retorne aqui o resultado da questão 3.
    return black_friday['User_ID'].unique().shape[0]
    pass 


# ## Questão 4
# 
# Quantos tipos de dados diferentes existem no dataset? Responda como um único escalar.

# In[14]:


black_friday.dtypes.nunique()


# In[15]:


def q4():
    # Retorne aqui o resultado da questão 4.
    return black_friday.dtypes.nunique()
    pass


# ## Questão 5
# 
# Qual porcentagem dos registros possui ao menos um valor null (`None`, `ǸaN` etc)? Responda como um único escalar entre 0 e 1.

# In[16]:


#é o total menos o valor dos registros que possuem valor dividido pelo o total
#ficando apenas os que nao possuem registro

(black_friday.shape[0] - black_friday.dropna().shape[0]) / black_friday.shape[0]


# In[17]:


def q5():
    # Retorne aqui o resultado da questão 5.
    return (black_friday.shape[0] - black_friday.dropna().shape[0]) / black_friday.shape[0]
    pass


# ## Questão 6
# 
# Quantos valores null existem na variável (coluna) com o maior número de null? Responda como um único escalar.

# In[18]:


black_friday.isna().sum().max()


# In[19]:


Q6=black_friday.isna().sum().max()


# In[20]:


def q6():
    # Retorne aqui o resultado da questão 6.
    return Q6
    pass


# ## Questão 7
# 
# Qual o valor mais frequente (sem contar nulls) em `Product_Category_3`? Responda como um único escalar.

# In[21]:


black_friday['Product_Category_3'].value_counts()


# In[22]:


valores=black_friday['Product_Category_3'].value_counts()


# In[23]:


Q7=16.0


# In[24]:


def q7():
    # Retorne aqui o resultado da questão 7.
    return Q7
    pass


# ## Questão 8
# 
# Qual a nova média da variável (coluna) `Purchase` após sua normalização? Responda como um único escalar.

# In[25]:


black_friday['Purchase']


# In[26]:


aux = black_friday['Purchase']


# In[27]:


normal = (aux - aux.min()) / (aux.max() - aux.min())


# In[28]:


Q8=normal.mean()


# In[29]:


Q8


# In[30]:


def q8():
    # Retorne aqui o resultado da questão 8.
    return Q8
    pass 


# ## Questão 9
# 
# Quantas ocorrências entre -1 e 1 inclusive existem da variáel `Purchase` após sua padronização? Responda como um único escalar.

# Normalizar seguindo a formula, signifia, z= (x - media)/(desvio padrão)

# In[31]:


DV=black_friday['Purchase'].std()


# In[32]:


M=black_friday['Purchase'].mean()


# In[33]:


q9=(black_friday['Purchase']-M)/DV


# In[34]:


q9


# In[35]:


def q9():
    # Retorne aqui o resultado da questão 9.
    DV=black_friday['Purchase'].std()
    M=black_friday['Purchase'].mean()
    q9=(black_friday['Purchase']-M)/DV
    return q9[(q9 >= -1) & (q9 <= 1)].shape[0]
    pass 


# ## Questão 10
# 
# Podemos afirmar que se uma observação é null em `Product_Category_2` ela também o é em `Product_Category_3`? Responda com um bool (`True`, `False`).

# In[38]:


n1 = black_friday['Product_Category_3'].isna().sum()


# In[41]:


n2 = black_friday['Product_Category_2'].isna().sum()


# In[42]:


def q10():
    n1 = black_friday[(black_friday['Product_Category_2'].isna() == True) 
    & (black_friday['Product_Category_3'].isna() == True)].shape[0]
    
    n2 = black_friday['Product_Category_2'].isna().sum()
    
    # Retorne aqui o resultado da questão 10.
    return bool (n2 == n1)
    pass 


# In[ ]:




