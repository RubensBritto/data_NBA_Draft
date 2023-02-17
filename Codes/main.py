import pandas as pd
import numpy as np
import sklearn
import lazypredict
from sklearn.model_selection import train_test_split
from lazypredict.Supervised import LazyClassifier
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.metrics import confusion_matrix



# name Pip venv - venvMaster2
# Activate venv Pip - venvMaster2\Scripts\Activate
# Run Code - python main.py

dados_2009_2021 = pd.read_csv("https://raw.githubusercontent.com/RubensBritto/data_NBA_Draft/main/Estatisticas%20Avancadas/CollegeBasketballPlayers2009-2021.csv")
dados_2022 = pd.read_csv("https://raw.githubusercontent.com/RubensBritto/data_NBA_Draft/main/Estatisticas%20Avancadas/CollegeBasketballPlayers2022.csv")
target_draft =pd.read_csv("https://raw.githubusercontent.com/RubensBritto/data_NBA_Draft/main/Estatisticas%20Avancadas/DraftedPlayers2009-2021_2.csv")


target_draft = target_draft.rename(columns={'YEAR': 'year'}) # Replace no nome da coluna

data = pd.concat([dados_2009_2021, dados_2022], axis=0, ignore_index=True) # Concatenar os dataSet

m1 = pd.merge(data,target_draft,how = 'left', on = ['player_name', 'year']) # Mesclar os dataSet de acordo com as colunas nome e year

m1.drop(["AFFILIATION", "TEAM", "year", "ROUND.1", "OVERALL",'Unnamed: 64', 'Unnamed: 65'], axis=1, inplace=True)

m1["ROUND"]  = m1["ROUND"].replace(np.nan, 0)
print(f'Unique key Round: {m1["ROUND"].unique()}')

m1.drop(["pick", "Rec Rank"], axis=1, inplace=True)
m1.drop(['player_name', 'team', 'conf','yr','ht','type','porpag', 'num'], axis=1, inplace=True)

for i in m1.columns:
  m1[i] = m1[i].replace(np.nan,0)
  
# a = m1.corr()
# a.to_csv("correlacao.csv")


#### Balanceamento ######

# X = m1.sample(n=5000) # Escolhendo de forma aleatori
X1 = m1[(m1["ROUND"]  == 1)]
X2 = m1[(m1["ROUND"]  == 2)]
X3 = m1[(m1["ROUND"]  == 0)]

X4 = X3.sample(n=1000, random_state = 33)


data_final = pd.concat([X1, X2, X4], axis=0, ignore_index=True) # Concatenar os dataSet
# data_final.to_csv("newDataSet.csv")

print(f'round = 1 : {len(X1)}')
print(f'round = 2 : {len(X2)}')
print(f'round = 0 : {len(X3)}')

######################################################################


# y = data_final["ROUND"]
# data_final.drop('ROUND', axis=1, inplace=True)
# X = data_final

# print(X)

### Normalização #####
# transformer = MinMaxScaler().fit(X) #Normalização com MinMax
# m2 = transformer.fit_transform(X)

# transformer = MaxAbsScaler().fit(X) # Normalização com MaxAbsScaler
# m2 = transformer.fit_transform(X)


# Usando tecnica de balanceamento -  Under-sampling
# rus = RandomUnderSampler()
# X_res, y_res = rus.fit_resample(x, y)
# print('Resampled Y %s' % Counter(y_res))
# print('Resampled X %s' % Counter(X_res))


# X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.33,random_state =50)

def calc_metrics(tp, tn, fp, fn):
  accuracy = (tp + tn)/(tp + fp + tn + fn)
  precision = (tp)/(tp + fp)
  recall = (tp) / (tp + fn)
  f1_score = 2 * (precision * recall) / (precision + recall)
  print(f'accuracy: {accuracy}')
  print(f'precision: {precision}')
  print(f'recall: {recall}')
  print(f'f1_score: {f1_score}')

# print('Resampled dataset shape %s' % Counter(y_res))
# print(f'Y-test {Counter(y_test)}')
# clf = LazyClassifier(verbose=0,ignore_warnings=True, custom_metric=None)
# models,predictions = clf.fit(X_train, X_test, y_train, y_test)

# print(models)

################### Wrapper Metodo - Forward Selection #########

# sfs = SFS(KNeighborsClassifier(n_neighbors=7),
#            k_features=53,
#            forward=True,
#            floating=True,
#            cv =5)

# sfs.fit(X, y)

# df_SFS_results = pd.DataFrame(sfs.subsets_).transpose()
# df_SFS_results.to_csv("output.csv")
###################################################################

  
################### Embedding Metodo #############################

# from sklearn.tree import DecisionTreeClassifier
# listTest = []
# clf = DecisionTreeClassifier(random_state=0, criterion='entropy').fit(X_train, y_train)
# print(clf.score(X_test, y_test))
# for i in zip(X.columns, clf.feature_importances_):
#   #if i[1] > 0.0:
#   listTest.append(i)
#   print(i)
  
# y_pred_cv_lr = clf.predict(X_test)
# cm_lr_cv = confusion_matrix(y_test, y_pred_cv_lr)

# data_embedding = pd.DataFrame(listTest)
# data_embedding.to_csv("output_embedding.csv")

###################################################################

############################### Encontra o K, para o KNN do metodo Wrapper ################################
# for i in range(2,20):
#   neigh = KNeighborsClassifier(n_neighbors=i)
#   neigh.fit(X_train,y_train)
#   y_pred = neigh.predict(X_test)
#   print(f'Aucaria :{metrics.accuracy_score(y_test,y_pred)} - k : {i}')
  
#############################################################################################################

############################################## Filter Metodo #########################################
# importances = data_final.drop('ROUND', axis=1).apply(lambda x: x.corr(data_final.ROUND))
# indices = np.argsort(importances)
# print(importances[indices])
# imptance_csv = pd.DataFrame(importances[indices])

# imptance_csv.to_csv("filter.csv")

#############################################################################################################

### Arquivo com as planilhas - https://drive.google.com/drive/u/1/folders/1tzGgis6TgzVcQdVr_vqzZw-ZZX8Hm9Xt

### Assistir para interpretar a matrix de confusão - https://www.youtube.com/watch?v=FMVXocEqvuA
