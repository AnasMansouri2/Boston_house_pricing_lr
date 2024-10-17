####################################
### Projet BOSTON HOUSING PRICES ###
####################################


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
import pickle



#=======================================================================================================#


### Chargement du Dataset à partir d'une URL (DS supprime de sklearn pour des raisons d'ethique):

url = "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
df = pd.read_csv(url)

#=======================================================================================================#

### Preparation du Dataset:

## Prise en main du Dataset:
print(df.head())
df.info()

## Changement du nom de la target variable:
df.rename(columns={'medv': 'price'}, inplace=True)

## Resume statistqiue:
df.describe()

"""
Informations sur les variables du Dataset:

        - CRIM taux de criminalité par habitant par ville
        - ZN proportion de terrains résidentiels zonés pour des lots de plus de 25 000 pieds carrés
        - INDUS proportion d'acres d'entreprises autres que de vente au détail par ville
        - CHAS Variable muette Charles River (= 1 si le territoire est bordé par la rivière ; 0 sinon)
        - NOX concentration d'oxydes nitriques (parties par 10 millions)
        - RM nombre moyen de pièces par logement
        - AGE proportion de logements occupés par leur propriétaire et construits avant 1940
        - DIS distances pondérées par rapport à cinq centres d'emploi de Boston
        - RAD indice d'accessibilité aux autoroutes radiales
        - TAX taux d'imposition sur la pleine valeur de la propriété pour 10 000 dollars
        - PTRATIO taux d'encadrement par ville
        - B 1000(Bk - 0,63)^2 où Bk est la proportion de Noirs par ville
        - LSTAT % de statut inférieur de la population
        - MEDV Valeur médiane des maisons occupées par leur propriétaire en milliers de dollars
"""
## Checker les missing values:
df.isnull().sum()

#=======================================================================================================#

### EDA:

## Correlations:
corr = df.corr()

# Heatmap avec seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1, linewidths=0.5, linecolor='white', cbar_kws={'shrink': 0.8})
plt.title('Matrice de correlation des variables', fontsize=16)
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# Focus sur quelques nuages de points interessants :
sns.scatterplot(data=df, x='crim', y='price')
sns.scatterplot(data=df, x='rm', y='price')
sns.scatterplot(data=df, x='chas', y='price')

""" 
Focus sur quelques variables:
- La variable CRIM est négativement correlee avec notre variable d'interet : le prix median d'un bien immobilier baisse lorsqu'il y'a une hausse du taux de criminalité dans le secteur.
- La variable RM est positivement correlee avec notre variable d'ineteret : en moyenne , plus on a de chambres plus le prix du bien augmente.
- La variable CHAS n'est pas correlee avec notre variable d'interet
"""

#=======================================================================================================#

### Preparation du df pour la modelisation:

## Separation des features en dependantes et independantes:
X = df.iloc[:,:-1]
y = df.iloc[:,-1]

## Train et Test Split : 
X_train,X_test,y_train,y_test =train_test_split(X,y,test_size=0.3,random_state=42)

## Standadisation du Dataset:
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Creation et importation du pickle file pour le scaler:
pickle.dump(scaler,open('scaling.pkl','wb'))


#=======================================================================================================#

### Entrainement du model :

## Importation du regresseur lineaire:
regression=LinearRegression()
regression.fit(X_train,y_train)

## Afficher les coefficients de la regression et la constante :
print(regression.coef_)
print(regression.intercept_)

#=======================================================================================================#

### Predictions avec les données de test:

reg_pred = regression.predict(X_test)
reg_pred

## Verification rapide avec un scatterplot de la coherence des predictions sur les donnes de test:
plt.scatter(y_test,reg_pred)

"""
Quasi linéarité du nuage de points : les prédictions faites sur les données de test sont assez bonnes
"""

## Calcul des residus de la regression:
residus = y_test-reg_pred

## Plot des residus:
sns.displot(residus,kind="kde")

"""
Il est possible d'assumer une distribution normale des residus malgre la presence certaine d'outliers qui genent la distribution (à partir de 10)
"""

## Metrics pour juger de la qualité de la regresison:

print(mean_absolute_error(y_test,reg_pred))
print(mean_squared_error(y_test,reg_pred))
print(np.sqrt(mean_squared_error(y_test,reg_pred)))

"""
Interpretations :
MAE = 3.16 unités , Cela signifie que les prédictions de prix de maison sont en moyenne à 3,160 dollars près de la valeur réelle.
RMSE = 4.64 unités , Cela signifie que l'erreur moyenne du modèle est d'environ 4,640 dollars. Le RMSE, étant un peu plus élevé que le MAE, indique qu'il pourrait y avoir quelques écarts plus importants dans les prédictions.
"""

# R carre et R carre ajuste:
r_carre = r2_score(y_test,reg_pred)
print(r_carre)
r_carre_ajuste = 1 - (1-r_carre)*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1)
print(r_carre_ajuste)

"""
Interpretations:
r_carre : 71.12% de la variance des prix des maisons est expliquée par notre modèle. En d'autres termes, le modèle capture bien la relation entre les variables explicatives et le prix des maisons.
r_acrre_ajuste : signifie que, après avoir ajusté pour le nombre de variables dans le modèle, environ 68.40% de la variance des prix est expliquée.
"""

#=======================================================================================================#

### Predictions sur de nouvelles donnees : 

# On souhaite utiliser la première ligne du df d'origine:
new_data =  df.drop(columns=['price']).iloc[0].values.reshape(1, -1)

# On scale les nouvelles données:
scaled_new_data = scaler.transform(new_data)

# Predire:
predicted_value = regression.predict(scaled_new_data)

# Afficher la prédiction
print(f"Prédiction du prix avec pur les nouvelles données: {predicted_value}")


#=======================================================================================================#

### Pickling du modele de reg lineaire:

pickle.dump(regression,open('regmodel.pkl','wb'))
pickled_model=pickle.load(open('regmodel.pkl','rb'))

## Prediction
pickled_model.predict(scaler.transform(new_data))