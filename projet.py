#Bolea Alexandra

from matplotlib import table
import pandas as pd
import numpy as np
import numpydoc
from utils import AbstractClassifier, getNthDict
import utils
import pydot
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import scipy.stats as stats
import math
from scipy.stats import norm
import projet
import sys
import pydot
from IPython.display import display
import os
from collections import defaultdict
from sklearn.naive_bayes import GaussianNB
from scipy.stats import chi2_contingency
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
import pydoc

data=pd.read_csv("data/heart.csv")
data.head()

utils.viewData(data)

discretise=utils.discretizeData(data)
utils.viewData(discretise,kde=False)

train=pd.read_csv("data/train.csv")
test=pd.read_csv("data/test.csv")
utils.viewData(train,kde=False) 

def getPrior(dataset):
    """
    Cette fonction calcule la probabilité a priori de la classe `target` = 1 dans le dataset 
    fourni, ainsi qu'un intervalle de confiance à 95 % pour la moyenne, en utilisant un score z de 1,96.
    
    Args:
        dataset (pandas.DataFrame): Un dataset contenant une colonne binaire nommée 'target'. 
            Cette colonne doit indiquer si chaque instance appartient à la classe 1 ou non 0.

    Returns:
        dict: Un dictionnaire contenant les clés suivantes :
            - 'estimation' (float) : La probabilité estimée de la classe 1.
            - 'min5pourcent' (float) : La borne inférieure de l'intervalle de confiance à 95 % pour la moyenne.
            - 'max5pourcent' (float) : La borne supérieure de l'intervalle de confiance à 95 % pour la moyenne. 
    """
    n = len(dataset)
    instances_true = dataset['target'].sum()
    proba = instances_true/n
    z = 1.96
    mean = dataset['target'].mean()
    std = dataset['target'].std()
    margin = z * std / math.sqrt(n)
    
    return {
         'estimation' : float(proba),
         'min5pourcent' : float(mean - margin),
         'max5pourcent' : float(mean + margin)
     }

projet.getPrior(train)
projet.getPrior(test) 
    
class APrioriClassifier(AbstractClassifier):
    """
        Implémente un classifieur basé sur la probabilité a priori.
        Ce classifieur utilise la probabilité a priori de la classe 1 (calculée avec la fonction `getPrior`) 
        pour prédire les classes des instances et évaluer ses performances sur un dataset donné.
    """
    def __init__(self, df):
        """
        Constructeur. Initialise le classifieur a priori.

        Args:
            df (pandas.DataFrame): Le dataset utilisé pour entraîner ou évaluer le modèle.
        """
        super().__init__()
          
    def estimClass(self, attrs):
        """
        Estime la classe d'une instance en fonction des attributs fournis.
        Cette méthode utilise la probabilité a priori calculée sur l'ensemble d'entraînement (`train`)
        pour prédire la classe de l'instance.

        Args:
            attrs (dict): Un dictionnaire contenant les attributs de l'instance.

        Returns:
            int: La classe prédite (1 pour l'évènement positif, 0 pour l'évènement négatif).
        """
        res = getPrior(train)
        if(res['estimation'] > 0.5): return 1
        else: return 0
    
    def statsOnDF(self, df):
        """
        Calcule des statistiques de performance du classifieur sur un dataset.
        Les statistiques incluent le nombre de vrais positifs (VP), vrais négatifs (VN), 
        faux positifs (FP), et faux négatifs (FN), ainsi que la précision et le rappel.

        Args:
            df (pandas.DataFrame): Un dataset contenant les instances à évaluer, avec une colonne 'target' 
            qui représente la classe réelle.

        Returns:
            dict: Un dictionnaire contenant les statistiques suivantes :
                - 'VP' (int) : Nombre de vrais positifs (prédit 1, vrai 1).
                - 'VN' (int) : Nombre de vrais négatifs (prédit 0, vrai 0).
                - 'FP' (int) : Nombre de faux positifs (prédit 1, vrai 0).
                - 'FN' (int) : Nombre de faux négatifs (prédit 0, vrai 1).
                - 'Précision' (float) : Précision du modèle (VP / (VP + FP)).
                - 'Rappel' (float) : Rappel du modèle (VP / (VP + FN)).
        """
        dic = {}
        VP = 0
        VN = 0
        FP = 0
        FN = 0
        
        for t in df.itertuples():
            dic = t._asdict()
            #print(f"ca={dic['ca']} oldpeak={dic['oldpeak']} target={dic['target']}")
            predicted_class = self.estimClass(dic)
            if dic.get('target') == 1 and predicted_class == 1:
                VP += 1
            elif dic.get('target') == 1 and predicted_class == 0:
                FN += 1
            elif dic.get('target') == 0 and predicted_class == 1:
                FP +=1
            elif dic.get('target') == 0 and predicted_class == 0:
                VN += 1
        
        
        precision = VP / (VP + FP) if (VP + FP) != 0 else 0
        recall = VP / (VP + FN) if (VP + FN) != 0 else 0
        
        return {
            'VP': VP,
            'VN': VN,
            'FP': FP,
            'FN': FN,
            'Précision': precision,
            'Rappel': recall
        }

cl=APrioriClassifier(train)
print("test en apprentissage : {}".format(cl.statsOnDF(train)))
print("test en validation: {}".format(cl.statsOnDF(test)))

def P2D_l(df, attr):
    """"
    Calcule la probabilité conditionnelle d'une valeur d'attribut donnée
    pour chaque classe cible dans le DataFrame.
    Cette fonction génère un dictionnaire où chaque clé correspond à une classe cible (`target`), 
    et chaque valeur est un sous-dictionnaire. Ce sous-dictionnaire associe les valeurs possibles 
    de l'attribut spécifié (`attr`) à leurs fréquences relatives (probabilités) conditionnées 
    par la classe cible.

    Args:
        df (pandas.DataFrame): Un DataFrame contenant une colonne `target` 
            (la variable cible) et une colonne correspondant à l'attribut spécifié.
        attr (str): Le nom de la colonne représentant l'attribut dont on veut 
            calculer la probabilité conditionnelle.

    Returns:
        dict: Un dictionnaire de probabilités conditionnelles sous la forme :
            {
                0: {
                    valeur_attribut_1: probabilité,
                    valeur_attribut_2: probabilité,
                    ...
                },
                1: {
                    valeur_attribut_1: probabilité,
                    valeur_attribut_2: probabilité,
                    ...
                }
            }
    """
    target_values = df['target'].unique() 
    attr_values = df[attr].unique()  
    
    prob_dict = {}

    for target in target_values:
        target_subset = df[df['target'] == target]
        
        freq = target_subset[attr].value_counts(normalize=True)
        
        prob_dict[int(target)] = {int(val): float(freq.get(val, 0)) for val in attr_values}
    
    return prob_dict
    
p_thal_given_target = P2D_l(train, 'thal')
print(p_thal_given_target)
print()
print(f"Dans la base train, la probabilité que thal=3 sachant que target=1 est {p_thal_given_target[1][3]:.10f}")

def P2D_p(df, attr):
    """
    Calcule la probabilité conditionnelle de chaque classe cible (`target`)
    pour chaque valeur d'un attribut donné dans un DataFrame.
    Cette fonction génère un dictionnaire où chaque clé correspond à une valeur unique 
    de l'attribut spécifié (`attr`), et chaque valeur est un sous-dictionnaire. Ce sous-dictionnaire 
    associe les classes cibles (`target`) à leurs fréquences relatives (probabilités) 
    conditionnées par la valeur de l'attribut.

    Args:
        df (pandas.DataFrame): Un DataFrame contenant une colonne `target` 
            (la variable cible) et une colonne correspondant à l'attribut spécifié.
        attr (str): Le nom de la colonne représentant l'attribut dont on veut 
            calculer la probabilité conditionnelle.

    Returns:
        dict: Un dictionnaire de probabilités conditionnelles sous la forme :
            {
                valeur_attribut_1: {
                    0: probabilité,
                    1: probabilité
                },
                valeur_attribut_2: {
                    0: probabilité,
                    1: probabilité
                },
                ...
            }
    """
    target_values = df['target'].unique() 
    attr_values = df[attr].unique()  
    
    prob_dict = {}

    for val in attr_values:
        attr_subset = df[df[attr] == val]
        
        freq = attr_subset['target'].value_counts(normalize=True)
        
        prob_dict[int(val)] = {int(t): float(freq.get(t, 0)) for t in target_values}
    
    return prob_dict

p_target_given_thal = P2D_p(train, 'thal')
print(p_target_given_thal)
print()
print(f"Dans la base train, la probabilité que target=1 sachant que thal=3 est {p_target_given_thal[3][1]:.10f}")

class ML2DClassifier(APrioriClassifier):
    """
    Implémente un classifieur bayésien simple basé sur la probabilité conditionnelle 
    d'une classe cible en fonction des valeurs d'un attribut donné.
    Ce classifieur hérite de la classe `APrioriClassifier` et utilise les probabilités conditionnelles 
    calculées à l'aide de la fonction `P2D_l`. Les prédictions sont basées sur la valeur de l'attribut 
    spécifié et les distributions conditionnelles des classes cibles.

    Args:
        APrioriClassifier (class): inheritance
    """
    def __init__(self, df, prop):
        """
        Initialise le classifieur.
        Attributs créés:
            - `self.prop`: Stocke le nom de l'attribut utilisé pour la classification.
            - `self.table`: Génère un tableau contenant les probabilités conditionnelles 
              pour chaque classe cible en fonction des valeurs de l'attribut, en utilisant la fonction `P2D_l`.

        Args:
            df (pandas.DataFrame): Le DataFrame d'entraînement contenant une colonne `target` (classe cible) 
                et l'attribut spécifié.
            prop (str): Le nom de l'attribut utilisé pour la classification.
        """
        super().__init__(df)
        self.prop = prop
        p = P2D_l(df, prop)
        self.table = np.array(list(p.items()))
       
    def estimClass(self, dict):
        """
        Prédit la classe cible pour une observation donnée.
        La méthode utilise les probabilités conditionnelles calculées dans `self.table` 
        pour déterminer la classe cible avec la plus grande probabilité. En cas d'égalité 
        des probabilités entre les classes, elle retourne 0 par défaut.


        Args:
            dict (dict): Un dictionnaire contenant les valeurs des attributs pour une observation.
                La clé correspondant à `self.prop` doit être présente.

        Raises:
            KeyError: Si le dictionnaire d'entrée ne contient pas la clé `self.prop`.

        Returns:
            int: La classe prédite (0 ou 1).
        """
        if self.prop not in dict:
            raise KeyError(f"The key '{self.prop}' is missing in the input dictionary.")
        
        attr = dict[self.prop]
        p = [0, 0]
        
        for i in range(len(self.table)):
            class_label, prob_dict = self.table[i]
            if attr in prob_dict:
                p[class_label] = prob_dict[attr]
                         
        if p[0] > p[1]:
            return 0
        elif p[0] == p[1]:
            return 0  
        else:
            return 1        
            
    
cl = ML2DClassifier(train,'thal')
for i in [0,1,2]:
    print("Estimation de la classe de l'individu {} par ML2DClassifier : {}".format(i, cl.estimClass(utils.getNthDict(train, i))))
    
print("test en apprentissage : {}".format(cl.statsOnDF(train)))
print("test en validation: {}".format(cl.statsOnDF(test)))

class MAP2DClassifier(APrioriClassifier):
    """
    Implémente un classifieur bayésien basé sur la probabilité a posteriori 
    d'une classe cible conditionnée par une valeur d'attribut.
    Ce classifieur hérite de la classe `APrioriClassifier` et utilise les probabilités 
    conditionnelles inversées, calculées avec la fonction `P2D_p`. Les prédictions sont 
    basées sur la valeur de l'attribut spécifié et les probabilités a posteriori des classes cibles.

    Args:
        APrioriClassifier (class): inherité
    """
    def __init__(self, df, prop):
        """
        Initialise le classifieur.
        Attributs créés:
            - `self.prop`: Stocke le nom de l'attribut utilisé pour la classification.
            - `self.table`: Une liste de tuples contenant les valeurs d'attribut et leurs probabilités 
            a posteriori pour chaque classe cible, générée avec la fonction `P2D_p`.

        Args:
            df (pandas.DataFrame): Le DataFrame d'entraînement contenant une colonne `target` 
                (classe cible) et une colonne représentant l'attribut spécifié.
            prop (str): Le nom de l'attribut utilisé pour la classification.
        """
        super().__init__(df)
        self.prop = prop
        p = P2D_p(df, prop)
        self.table = list(p.items())
       
    def estimClass(self, dict):
        """
        Prédit la classe cible pour une observation donnée.
        La méthode utilise les probabilités a posteriori contenues dans `self.table` pour déterminer 
        la classe cible ayant la probabilité maximale. En cas d'égalité des probabilités entre les classes, 
        elle retourne 0 par défaut.

        Args:
            dict (dict): Un dictionnaire contenant les valeurs des attributs pour une observation.
                La clé correspondant à `self.prop` doit être présente.

        Raises:
            KeyError: Si le dictionnaire d'entrée ne contient pas la clé `self.prop`

        Returns:
            int: La classe prédite (0 ou 1).
        """
        if self.prop not in dict:
            raise KeyError(f"The key '{self.prop}' is missing in the input dictionary.")
        
        attr = dict[self.prop]
        p = [0, 0]
        
        for entry in self.table:
            if entry[0] == attr:  
                prob_dict = entry[1]  
                p[0] = prob_dict[0]
                p[1] = prob_dict[1] 
             
        if p[0] > p[1]:
            return 0
        elif p[0] == p[1]:
            return 0  
        else:
            return 1    
        

cl=projet.MAP2DClassifier(train,"thal") # cette ligne appelle projet.P2Dp(train,"thal")
for i in [0,1,2]:
    print("Estimation de la classe de l'individu {} par MAP2DClasssifer) : {}".format(i,cl.estimClass(utils.getNthDict(train,i))))
    
print("test en apprentissage : {}".format(cl.statsOnDF(train)))
print("test en validation: {}".format(cl.statsOnDF(test)))

def nbParams(df, attrs=None):
    """"
    Cette fonction évalue le nombre d'attributs et le produit cartésien des valeurs distinctes 
    de chaque attribut pour déterminer la quantité de mémoire requise pour stocker toutes 
    les combinaisons possibles de valeurs d'attributs. Elle suppose que chaque combinaison 
    de valeurs est représentée par un entier de 64 bits (8 octets).

    Args:
        df (pandas.DataFrame): Le DataFrame contenant les données.
        attrs (list, optional): Liste des noms des colonnes (attributs) pour lesquels 
            les valeurs distinctes seront prises en compte. Si `None` (par défaut), 
            tous les attributs du DataFrame sont utilisés.
    
    Returns:
        None: Affiche simplement le nombre d'octets nécessaires pour stocker toutes 
              les combinaisons possibles des attributs spécifiés.
    """
    if attrs is None:
        attrs = df.columns.tolist()
        
    num_attrs = len(attrs)
    
    num_rows = df.shape[0]
    
    total_bytes = 0
    cart_prod = 1
    
    for attr in attrs:
        num_distinct_values = df[attr].nunique()
        cart_prod = cart_prod * num_distinct_values
        
    total_bytes = 8 * cart_prod 
    
    print(f"{num_attrs} variables: {total_bytes} octets")

projet.nbParams(train,['target'])
projet.nbParams(train,['target','thal'])
projet.nbParams(train,['target','age'])
projet.nbParams(train,['target','age','thal','sex','exang'])
projet.nbParams(train,['target','age','thal','sex','exang','slope','ca','chol'])
projet.nbParams(train) # seul résultat visible en sortie de cellule

def nbParamsIndep(df, attrs=None):
    """
    Calcule le nombre d'octets nécessaires pour représenter les valeurs distinctes de chaque 
    attribut spécifié dans un DataFrame, en supposant que chaque valeur distincte est 
    représentée par un entier de 64 bits (8 octets).

    Args:
        df (pandas.DataFrame): Le DataFrame contenant les données.
        attrs (list, optional): Liste des noms des colonnes (attributs) pour lesquels 
            les valeurs distinctes seront prises en compte. Si `None` (par défaut), 
            tous les attributs du DataFrame sont utilisés.

    Returns:
        None: Affiche simplement le nombre d'octets nécessaires pour représenter 
              les valeurs distinctes de chaque attribut.
    """
    if attrs is None:
       attrs = df.columns.tolist()
        
    num_attrs = len(attrs)
    
    num_rows = df.shape[0]
    
    total_bytes = 0
    
    for attr in attrs:
        num_distinct_values = df[attr].nunique()
        total_bytes += num_distinct_values*8
    
    print(f"{num_attrs} variables: {total_bytes} octets")
    
projet.nbParamsIndep(train[['target']])
projet.nbParamsIndep(train[['target','thal']])
projet.nbParamsIndep(train[['target','age']])
projet.nbParamsIndep(train[['target','age','thal','sex','exang']])
projet.nbParamsIndep(train[['target','age','thal','sex','exang','slope','ca','chol']])
projet.nbParamsIndep(train) # seul résultat visible en sortie de cellule

def drawNaiveBayes(df, attr):
    """
    Génère un graphique représentant un modèle de Naive Bayes basé sur les relations entre 
    l'attribut cible et les autres attributs du DataFrame.
    La fonction crée une chaîne de caractères qui décrit un graphe de type `graphviz`, 
    où chaque attribut (à l'exception de l'attribut cible) est relié à l'attribut cible. 
    Cela correspond à un modèle de Naive Bayes où la variable cible est conditionnée par 
    les autres attributs.

    Args:
        df (pandas.DataFrame): Le DataFrame contenant les données.
        attr (str): Le nom de l'attribut cible, qui sera utilisé comme variable dépendante 
            dans le modèle Naive Bayes. Tous les autres attributs seront considérés comme 
            variables indépendantes.

    Returns:
        utils.drawGraph: Fonction qui génère et affiche un graphique représentant le modèle 
                         de Naive Bayes.
    """
    attrs = [col for col in df.columns if col != attr]
    
    graph_str = ""
    
    for a in attrs:
        graph_str += f"{attr}->{a};"
    
    return utils.drawGraph(graph_str)


projet.drawNaiveBayes(train,"target")

#Je n'ai pas travaille dans un JupiterNotebook,j'ai realise ca a mi-chemin, donc j'affiche le graph par hardcoder le chemin :)
graph = drawNaiveBayes(train,"target")  
graph_file = "naive_bayes_graph.png"
with open(graph_file, "wb") as f:
    f.write(graph.data)

os.startfile(graph_file)

def nbParamsNaiveBayes(df, attr_given ,attrs = None):
    """
    Calcule le nombre d'octets nécessaires pour représenter les paramètres d'un modèle Naive Bayes, 
    basé sur les attributs spécifiés dans un DataFrame.
    Cette fonction estime la mémoire requise pour stocker les paramètres d'un modèle Naive Bayes. 
    Le modèle Naive Bayes considère l'attribut cible comme étant conditionné par les autres 
    attributs (variables indépendantes). Le calcul est effectué en tenant compte des valeurs 
    distinctes de chaque attribut, avec l'hypothèse que chaque combinaison de valeurs nécessite 
    8 octets pour être stockée.

    Args:
        df (pandas.DataFrame): Le DataFrame contenant les données.
        attr_given (str): Le nom de l'attribut cible (conditionné par les autres attributs dans Naive Bayes).
        attrs (list, optional): Liste des noms des colonnes (attributs) pour lesquels les valeurs 
            distinctes seront prises en compte. Si `None` (par défaut), tous les attributs du DataFrame 
            sont utilisés sauf l'attribut cible `attr_given`.
            
    Returns:
        None: Affiche simplement le nombre d'octets nécessaires pour stocker les paramètres du modèle 
            Naive Bayes.
    """
    if attrs is None:
       attrs = df.columns.tolist()
    
    num_distinct_values = df[attr_given].nunique()
    total_bytes = num_distinct_values
    
    for attr in attrs:
        if attr != attr_given:
            n = len(df[attr].unique())
            total_bytes += num_distinct_values*n
        
    mem = total_bytes*8
    
    print(f"{len(attrs)} variables: {mem} octets")

print("bayes:")
projet.nbParamsNaiveBayes(train,'target',[])
projet.nbParamsNaiveBayes(train,'target',['target','thal'])
projet.nbParamsNaiveBayes(train,'target',['target','age'])
projet.nbParamsNaiveBayes(train,'target',['target','age','thal','sex','exang'])
projet.nbParamsNaiveBayes(train,'target',['target','age','thal','sex','exang','slope','ca','chol'])
projet.nbParamsNaiveBayes(train,'target') # seul résultat visible en sortie de cellule

class MLNaiveBayesClassifier(APrioriClassifier):
    """Classificateur Naive Bayes basé sur un modèle de maximum de vraisemblance (ML).
    Cette classe implémente un classificateur Naive Bayes utilisant un modèle de maximum 
    de vraisemblance pour estimer les probabilités conditionnelles entre les attributs et 
    la classe cible. Les probabilités sont calculées à partir des données d'entraînement en 
    utilisant la méthode de fréquence des valeurs.
    
    Args:
        APrioriClassifier (class): herité
    """
    def __init__(self, df):
        """Initialise le classificateur Naive Bayes en calculant les probabilités conditionnelles 
        entre chaque attribut et la variable cible.
        Attributs:
        
        self.table (numpy.ndarray): Une table contenant les probabilités conditionnelles 
            entre chaque attribut et la classe cible. La table est structurée sous forme 
            de tuples `(attribut, valeur de l'attribut, probabilités conditionnelles)`.

        Args:
            df (pandas.DataFrame): 
        """
        super().__init__(df)
        self.table = []  

        for attr in df:
            if attr != 'target': 
                p = P2D_l(df, attr)  
                for value, prob in p.items():
                    self.table.append((attr, value, prob)) 

        self.table = np.array(self.table, dtype=object)
                           
    def estimProbas(self, dict):
        """Estime les probabilités conditionnelles pour chaque classe cible en fonction 
        des attributs donnés.
        Cette méthode applique le théorème de Bayes pour estimer les probabilités de chaque 
        classe cible en fonction des attributs fournis dans le dictionnaire `dict`.

        Args:
            dict (dict): Un dictionnaire contenant les attributs et leurs valeurs 
                pour lesquels les probabilités doivent être estimées.

        Returns:
            dict: Un dictionnaire contenant les probabilités estimées pour chaque classe cible 
                (0 ou 1).
        """
        p = {0:1, 1:1}
        
        for attr, attr_value in dict.items():
            for entry in self.table:
                if entry[0] == attr:
                    target = entry[1]
                    prob_dict = entry[2]
                    if attr_value in prob_dict:
                        p[target] *= prob_dict[attr_value]
                       
                               
        return p
                    
    def estimClass(self, dict):
        """Estime la classe cible en fonction des attributs fournis.
        Cette méthode utilise les probabilités estimées par `estimProbas` pour prédire la classe 
        la plus probable. Si les probabilités pour chaque classe sont égales, la classe 0 est 
        retournée par défaut.

        Args:
            dict (dict): Un dictionnaire contenant les attributs et leurs valeurs.

        Returns:
            int: La classe prédite (0 ou 1).
        """
        probas = self.estimProbas(dict)
        return int(max(probas, key=probas.get))

    
cl=projet.MLNaiveBayesClassifier(train)
for i in [0,1,2]:
    print(f"Estimation de la proba de l'individu {i} par MLNaiveBayesClassifier : {cl.estimProbas(utils.getNthDict(train,i))}")
    print(f"Estimation de la classe de l'individu {i} par MLNaiveBayesClassifier : {cl.estimClass(utils.getNthDict(train,i))}")
    print("------")
print(f"test en apprentissage : {cl.statsOnDF(train)}")
print(f"test en validation : {cl.statsOnDF(test)}")  
 
class MAPNaiveBayesClassifier(APrioriClassifier):
    """
    Classificateur Naive Bayes basé sur l'estimation du maximum a posteriori (MAP).
    Cette classe implémente un classificateur Naive Bayes qui utilise l'estimation du 
    maximum a posteriori pour prédire la classe la plus probable. L'estimation a posteriori 
    combine les probabilités conditionnelles des attributs (calculées par `MLNaiveBayesClassifier`) 
    avec les probabilités a priori des classes (calculées directement à partir de la distribution 
    des classes dans les données d'entraînement).

    Args:
        APrioriClassifier (classe): heritage
    """
    def __init__(self, df):
        """Initialise le classificateur Naive Bayes MAP en calculant les probabilités a priori des 
        classes et en créant une instance de `MLNaiveBayesClassifier` pour estimer les probabilités 
        conditionnelles des attributs.
        
        Attributes:
        pclass (dict): Un dictionnaire contenant les probabilités a priori des classes cibles 
            (0 ou 1) calculées à partir de la distribution des classes dans les données.
        prob (MLNaiveBayesClassifier): Une instance de `MLNaiveBayesClassifier` utilisée 
            pour calculer les probabilités conditionnelles des attributs.

        Args:
            df (pandas.DataFrame): Le DataFrame contenant les données.
        """
        super().__init__(df) 
        
        self.pclass = df['target'].value_counts().to_dict()
        tot = len(df)
        
        self.pclass = {k: v / tot for k, v in self.pclass.items()}    
        
        self.prob = projet.MLNaiveBayesClassifier(df) 
        
    def estimProbas(self, dict):
        """
        Estime les probabilités a posteriori des classes pour un ensemble donné d'attributs.
        Cette méthode applique le théorème de Bayes pour calculer les probabilités a posteriori 
        en multipliant les probabilités conditionnelles (obtenues via `MLNaiveBayesClassifier`) 
        par les probabilités a priori des classes, puis en normalisant les résultats.

        Args:
            dict (dict): Un dictionnaire contenant les attributs et leurs valeurs pour lesquels 
                les probabilités doivent être estimées.

        Returns:
             dict: Un dictionnaire contenant les probabilités a posteriori pour chaque classe (0 ou 1).
        """
        p = self.prob.estimProbas(dict)
        for c in p:
           p[c] = p[c] * self.pclass[c] 
           
        total_prob = sum(p.values())
        for c in p:
            p[c] /= total_prob
                                
        return p
        
    
    def estimClass(self, dict):
        """Estime la classe cible en fonction des attributs fournis, en utilisant les probabilités 
        a posteriori estimées par `estimProbas`.
        Cette méthode prédit la classe ayant la probabilité a posteriori la plus élevée

        Args:
            dict (dict): Un dictionnaire contenant les attributs et leurs valeurs.

        Returns:
            int: La classe prédite (0 ou 1).
        """
        probas = self.estimProbas(dict)
        return int(max(probas, key=probas.get))
    
cl=projet.MAPNaiveBayesClassifier(train)
for i in [0,1,2]:
    print(f"Estimation de la proba de l'individu {i} par MAPNaiveBayesClassifier : {cl.estimProbas(utils.getNthDict(train,i))}")
    print(f"Estimation de la classe de l'individu {i} par MAPaiveBayesClassifier : {cl.estimClass(utils.getNthDict(train,i))}")
    print("------")
print(f"test en apprentissage : {cl.statsOnDF(train)}")
print(f"test en validation : {cl.statsOnDF(test)}")
    
def isIndepFromTarget(df, attr, x):
    """
    Vérifie si l'attribut spécifié est indépendant de la variable cible (target) 
    en utilisant le test du chi2 d'indépendance.
    Cette fonction utilise le test du chi2 pour évaluer si l'attribut spécifié 
    est statistiquement indépendant de la variable cible. Si la p-valeur du test 
    est supérieure à un seuil donné `x`, alors l'attribut est considéré comme indépendant 
    de la variable cible, sinon l'attribut est considéré comme dépendant.

    Args:
        df (pandas.DataFrame): Le DataFrame contenant les données, avec une colonne 
                               'target' représentant la variable cible.
        attr (str): Le nom de la colonne (attribut) à tester pour l'indépendance par rapport à 
                    la variable cible 'target'.
        x (float): Le seuil de la p-valeur pour déterminer si l'attribut est indépendant 
                  de la variable cible. Si la p-valeur est supérieure à `x`, l'attribut 
                  est considéré comme indépendant.

    Returns:
        bool: Retourne `True` si l'attribut est indépendant de la variable cible 
              (p-valeur > x), sinon retourne `False`.
    
    """
    contingency_table = pd.crosstab(df[attr], df['target'])
    
    chi2, p_val, _, _ = chi2_contingency(contingency_table)
    
    return p_val > x


for attr in train.keys():
    if attr!='target':
        print(f"target independant de {attr} ? {'YES' if projet.isIndepFromTarget(train,attr,0.01) else 'no'}")
        
class ReducedMLNaiveBayesClassifier(MLNaiveBayesClassifier):
    """
    Un classifieur Naïve Bayes réduit qui sélectionne automatiquement les 
    caractéristiques les plus pertinentes en fonction d'un seuil de dépendance 
    statistique par rapport à la variable cible 'target'. Utilise ensuite un modèle 
    Naïve Bayes gaussien pour effectuer la classification.
    
    Args:
        MLNaiveBayesClassifier (class): heritage
    """
    def __init__(self, df, threshold):
        """
        Attributes:
        threshold (float): Seuil de p-valeur pour déterminer l'indépendance statistique.
        selected (list): Liste des attributs sélectionnés après application du seuil.
        df (pandas.DataFrame): Le DataFrame contenant les données à classer.
        model (GaussianNB): Le modèle Naïve Bayes gaussien entraîné avec les caractéristiques sélectionnées.

        Args:
            df (pandas.DataFrame): Le DataFrame contenant les données avec la colonne 'target'.
            threshold (float): Le seuil de p-valeur pour déterminer la sélection des attributs.
        """
        super().__init__(df)
        self.threshold = threshold
        self.selected = self.select_feature(df, threshold)
        self.df = df
        
        self.model = GaussianNB()
        self.model.fit(df[self.selected], df['target'])
        
    def select_feature(self, df, threshold):
        """Sélectionne les attributs qui ne sont pas indépendants de la variable cible 
        en fonction du seuil de p-valeur.
        
        Args:
            df (pandas.DataFrame): Le DataFrame contenant les données.
            threshold (float): Seuil de p-valeur pour déterminer l'indépendance.

        Returns:
            list: Liste des attributs sélectionnés qui ne sont pas indépendants de la cible.
        """
        selected = []
        for attr in df.columns:
            if attr != 'target' and not isIndepFromTarget(df, attr, threshold):
                selected.append(attr)
        return selected
        
    def estimProbas(self, dict):
        """Calcule les probabilités estimées pour chaque classe en fonction des attributs sélectionnés.

        Args:
            dict (dict): Dictionnaire des attributs et de leurs valeurs à classer.

        Returns:
            dict: Dictionnaire des probabilités estimées pour chaque classe.
        """
        filtered_dict = {key: value for key, value in dict.items() if key in self.selected}
        p = super().estimProbas(filtered_dict)
        return p
    
    def estimClass(self, dict):
        """Prédit la classe en fonction des probabilités estimées, en utilisant uniquement 
        les attributs sélectionnés.

        Args:
            dict (dict): Dictionnaire des attributs et de leurs valeurs à classer.

        Returns:
            int: La classe prédite (0 ou 1).
        """
        filtered_dict = {key: value for key, value in dict.items() if key in self.selected}
        return super().estimClass(filtered_dict)

    def draw(self):
        """Génère un graphe représentant les dépendances entre la variable cible et les 
        attributs sélectionnés.

        Returns:
            Graph: Le graphique généré représentant les relations entre 'target' et les attributs.
        """
        graph_str = ""
    
        for a in self.selected:
            if(a != 'target'):
                graph_str += f"{attr}->{a};"
    
        return utils.drawGraph(graph_str)

cl=projet.ReducedMLNaiveBayesClassifier(train,0.05)
cl.draw()   

graph = cl.draw()  
graph_file = "pic.png"
with open(graph_file, "wb") as f:
    f.write(graph.data)

os.startfile(graph_file) 
       
for i in [0,1,2]:
    print(f"Estimation de la proba de l'individu {i} par ReducedMLNaiveBayesClassifier : {cl.estimProbas(utils.getNthDict(train,i))}")
    print(f"Estimation de la classe de l'individu {i} par ReducedMLNaiveBayesClassifier : {cl.estimClass(utils.getNthDict(train,i))}")
    print("------")
print(f"test en apprentissage : {cl.statsOnDF(train)}")
print(f"test en validation : {cl.statsOnDF(test)}")

class ReducedMAPNaiveBayesClassifier(MAPNaiveBayesClassifier):
    """Un classifieur Naïve Bayes réduit basé sur l'approche MAP (Maximum A Posteriori) 
    qui sélectionne automatiquement les caractéristiques les plus pertinentes en fonction 
    d'un seuil de dépendance statistique par rapport à la variable cible 'target'. 
    Utilise ensuite un modèle Naïve Bayes gaussien pour effectuer la classification.

    Args:
        MAPNaiveBayesClassifier (class): _heritage
    """
    def __init__(self, df, threshold):
        """Attributes:
            threshold (float): Seuil de p-valeur pour déterminer l'indépendance statistique.
            selected (list): Liste des attributs sélectionnés après application du seuil.
                df (pandas.DataFrame): Le DataFrame contenant les données à classer.
                model (GaussianNB): Le modèle Naïve Bayes gaussien entraîné avec les caractéristiques sélectionnées.

        Args:
            df (pandas.DataFrame): Le DataFrame contenant les données avec la colonne 'target'.
            threshold (float): Le seuil de p-valeur pour déterminer la sélection des attributs.
        """
        super().__init__(df)
        self.threshold = threshold
        self.selected = self.select_feature(df, threshold)
        self.df = df
        
        self.model = GaussianNB()
        self.model.fit(df[self.selected], df['target'])
        
    def select_feature(self, df, threshold):
        """Sélectionne les attributs qui ne sont pas indépendants de la variable cible 
        en fonction du seuil de p-valeur.

        Args:
            df (pandas.DataFrame): Le DataFrame contenant les données.
            threshold (float): Seuil de p-valeur pour déterminer l'indépendance.

        Returns:
            list: Liste des attributs sélectionnés qui ne sont pas indépendants de la cible.
        """
        selected = []
        for attr in df.columns:
            if attr != 'target' and not isIndepFromTarget(df, attr, threshold):
                selected.append(attr)
        return selected
        
    def estimProbas(self, dict):
        """
        Calcule les probabilités estimées pour chaque classe en fonction des attributs sélectionnés.

        Args:
            dict (dict): Dictionnaire des attributs et de leurs valeurs à classer.

        Returns:
            dict: Dictionnaire des probabilités estimées pour chaque classe.
        """
        filtered_dict = {key: value for key, value in dict.items() if key in self.selected}
        p = super().estimProbas(filtered_dict)
        return p
    
    def estimClass(self, dict):
        """
        Prédit la classe en fonction des probabilités estimées, en utilisant uniquement 
        les attributs sélectionnés.

        Args:
            dict (dict): Dictionnaire des attributs et de leurs valeurs à classer.

        Returns:
            int: La classe prédite (0 ou 1).
        """
        filtered_dict = {key: value for key, value in dict.items() if key in self.selected}
        return super().estimClass(filtered_dict)

    def draw(self):
        """
        Génère un graphe représentant les dépendances entre la variable cible et les 
        attributs sélectionnés.

        Returns:
            Graph: Le graphique généré représentant les relations entre 'target' et les attributs.
        """
        graph_str = ""
    
        for a in self.selected:
            if(a != 'target'):
                graph_str += f"{attr}->{a};"
    
        return utils.drawGraph(graph_str)
    
cl=projet.ReducedMAPNaiveBayesClassifier(train,0.01)
cl.draw()

graph = cl.draw()  
graph_file = "pic1.png"
with open(graph_file, "wb") as f:
    f.write(graph.data)

os.startfile(graph_file)

for i in [0,1,2]:
    print(f"Estimation de la proba de l'individu {i} par ReducedMAPNaiveBayesClassifier : {cl.estimProbas(utils.getNthDict(train,i))}")
    print(f"Estimation de la classe de l'individu {i} par ReducedMAPNaiveBayesClassifier : {cl.estimClass(utils.getNthDict(train,i))}")
    print("------")
print(f"test en apprentissage : {cl.statsOnDF(train)}")
print(f"test en validation : {cl.statsOnDF(test)}")

def mapClassifiers(classifiers, data):
    """
    Affiche un graphique de dispersion (scatter plot) des classificateurs en fonction de leur précision et rappel. 
    La fonction permet de comparer les performances de différents classificateurs sur un même jeu de données 
    en utilisant les mesures de précision et de rappel.

    Args:
        classifiers (dict): Un dictionnaire où les clés sont les noms des classificateurs (ou leurs identifiants) 
                             et les valeurs sont des objets de type classifieur ayant une méthode `statsOnDF()`.
                             Chaque classifieur doit être capable de calculer les statistiques de précision et de rappel.
        data (pandas.DataFrame): Le jeu de données sur lequel les classificateurs sont évalués. Il doit contenir 
                                  une colonne `target` représentant la variable cible pour l'évaluation des performances.

    Returns:
        None: Affiche un graphique de dispersion avec les résultats de précision et rappel des classificateurs.
    """
    precision = []
    rappel = []
    labels = []

    for key, classifier in classifiers.items():
        stats = classifier.statsOnDF(data)

        prec = stats["Précision"]
        rapp = stats["Rappel"]
        
        precision.append(prec)
        rappel.append(rapp)
        labels.append(key) 

    plt.figure(figsize=(8, 6))
    plt.scatter(precision, rappel, color='red', marker='x', label="Classifiers")

    for i, label in enumerate(labels):
        plt.text(precision[i], rappel[i], f'{label}', fontsize=10, ha='right')

    # Add labels, grid, and title
    plt.xlabel("Precision")
    plt.ylabel("Rappel")
    plt.title("Classifier Performance: Precision vs Recall")
    plt.grid(True)
    plt.show()

    
projet.mapClassifiers({"1":projet.APrioriClassifier(train),
"2":projet.ML2DClassifier(train,"exang"),
"3":projet.MAP2DClassifier(train,"exang"),
"4":projet.MAPNaiveBayesClassifier(train),
"5":projet.MLNaiveBayesClassifier(train),
"6":projet.ReducedMAPNaiveBayesClassifier(train,0.01),
"7":projet.ReducedMLNaiveBayesClassifier(train,0.01),
},train)

projet.mapClassifiers({"1":projet.APrioriClassifier(train),
"2":projet.ML2DClassifier(train,"exang"),
"3":projet.MAP2DClassifier(train,"exang"),
"4":projet.MAPNaiveBayesClassifier(train),
"5":projet.MLNaiveBayesClassifier(train),
"6":projet.ReducedMAPNaiveBayesClassifier(train,0.01),
"7":projet.ReducedMLNaiveBayesClassifier(train,0.01),
},test)

def MutualInformation(df, x, y):
        """
        Calcule l'information mutuelle entre deux variables x et y.

        Args:
            df (pd.DataFrame): Le DataFrame contenant les données.
            x (str): Le nom de la première variable.
            y (str): Le nom de la deuxième variable.

        Returns:
            float: L'information mutuelle entre x et y.
        """
        joint_distribution = pd.crosstab(df[x], df[y], normalize=True)
        px = joint_distribution.sum(axis=1)
        py = joint_distribution.sum(axis=0)
        mutual_info = 0.0

        for i in joint_distribution.index:
            for j in joint_distribution.columns:
                pxy = joint_distribution.loc[i, j]
                if pxy > 0:
                    mutual_info += pxy * np.log2(pxy / (px[i] * py[j]))

        return mutual_info
    
for attr in train.keys():
    if attr!='target':
        print(f"target->{attr:10} : {projet.MutualInformation(train,'target',attr):5.7f}")
        
def ConditionalMutualInformation(df, x, y, z):
        """
        Calcule l'information mutuelle conditionnelle entre x et y, sachant z.

        Args:
            df (pd.DataFrame): Le DataFrame contenant les données.
            x (str): Le nom de la première variable.
            y (str): Le nom de la deuxième variable.
            z (str): Le nom de la variable conditionnelle.

        Returns:
            float: L'information mutuelle conditionnelle entre x et y sachant z.
        """
        conditional_mutual_info = 0.0
        
        for z_val in df[z].unique():
            sub_df = df[df[z] == z_val]
            pz = len(sub_df) / len(df)

            joint_distribution = pd.crosstab(sub_df[x], sub_df[y], normalize=True)
            pxz = joint_distribution.sum(axis=1)
            pyz = joint_distribution.sum(axis=0)

            for i in joint_distribution.index:
                for j in joint_distribution.columns:
                    pxyz = joint_distribution.loc[i, j]
                    if pxyz > 0:
                        conditional_mutual_info += pz * pxyz * np.log2(pxyz / (pxz[i] * pyz[j]))

        return conditional_mutual_info
    
cmis=np.array([[0 if x==y else projet.ConditionalMutualInformation(train,x,y,"target")
for x in train.keys() if x!="target"]
for y in train.keys() if y!="target"])
print(cmis[0:5,0:5]) # on affiche qu'une partie 5x5 de la matrice

def MeanForSymetricWeights(a):
        """
        Calcule la moyenne des poids pour une matrice symétrique de diagonale nulle.

        Args:
            a (np.ndarray): La matrice symétrique de diagonale nulle.

        Returns:
            float: La moyenne des poids.
        """
        n = a.shape[0]
        total = np.sum(a) / 2  # On divise par 2 car chaque poids est compté deux fois dans une matrice symétrique
        num_elements = n * (n - 1) / 2  # Nombre d'éléments hors diagonale dans une matrice symétrique
        return total / num_elements
    
def SimplifyConditionalMutualInformationMatrix(a):
        """
        Annule toutes les valeurs plus petites que la moyenne dans une matrice symétrique de diagonale nulle.

        Args:
            a (np.ndarray): La matrice symétrique de diagonale nulle.

        Returns:
            np.ndarray: La matrice simplifiée.
        """
        mean_weight = MeanForSymetricWeights(a)
        a[a < mean_weight] = 0
        return a
   
print(projet.MeanForSymetricWeights(cmis))

projet.SimplifyConditionalMutualInformationMatrix(cmis)
print(cmis[0:5,0:5])

def Kruskal(df, a):
        """
        Implémente l'algorithme de Kruskal pour trouver l'arbre de poids maximal.

        Args:
            df (pd.DataFrame): Le DataFrame contenant les données.
            a (np.ndarray): La matrice des poids (symétrique et simplifiée).

        Returns:
            list: Liste des arcs sous forme de triplets (noeud1, noeud2, poids).
        """
        from scipy.sparse.csgraph import minimum_spanning_tree

        edges = []
        attributes = list(df.columns[df.columns != 'target'])

        for i in range(a.shape[0]):
            for j in range(i + 1, a.shape[1]):
                if a[i, j] > 0:
                    edges.append((attributes[i], attributes[j], a[i, j]))

        edges = sorted(edges, key=lambda x: x[2], reverse=True)
        
        parent = {attr: attr for attr in attributes}

        def find(x):
            while parent[x] != x:
                x = parent[x]
            return x

        def union(x, y):
            root_x = find(x)
            root_y = find(y)
            if root_x != root_y:
                parent[root_y] = root_x

        result = []

        for u, v, weight in edges:
            if find(u) != find(v):
                union(u, v)
                result.append((u, v, weight))

        return result

liste_arcs=projet.Kruskal(train,cmis)
print(liste_arcs)

def ConnexSets(list_arcs):
        """
        Trouve les ensembles d'attributs connectés dans un graphe donné par une liste d'arcs.

        Args:
            list_arcs (list): Liste des arcs sous forme de triplets (noeud1, noeud2, poids).

        Returns:
            list: Liste d'ensembles d'attributs connectés.
        """
        from collections import defaultdict

        graph = defaultdict(set)

        for u, v, _ in list_arcs:
            graph[u].add(v)
            graph[v].add(u)

        visited = set()

        def dfs(node, component):
            visited.add(node)
            component.add(node)
            for neighbor in graph[node]:
                if neighbor not in visited:
                    dfs(neighbor, component)
                    
        components = []

        for node in graph:
            if node not in visited:
                component = set()
                dfs(node, component)
                components.append(component)

        return components

def OrientConnexSets(df, arcs, classe):
        """
        Oriente les ensembles connexes d'attributs en choisissant une racine pour chaque ensemble.

        Args:
            df (pd.DataFrame): Le DataFrame contenant les données.
            arcs (list): Liste des arcs sous forme de triplets (noeud1, noeud2, poids).
            classe (str): Le nom de la variable cible.

        Returns:
            list: Liste des arcs orientés.
        """
        connex_sets = ConnexSets(arcs)
        mutual_info = {attr: MutualInformation(df, classe, attr) for attr in df.columns if attr != classe}

        oriented_arcs = []

        for component in connex_sets:
            root = max(component, key=lambda attr: mutual_info[attr])
            
            visited = set()
            stack = [root]

            while stack:
                node = stack.pop()
                visited.add(node)

                for u, v, _ in arcs:
                    if u == node and v not in visited:
                        oriented_arcs.append((node, v))
                        stack.append(v)
                    elif v == node and u not in visited:
                        oriented_arcs.append((node, u))
                        stack.append(u)

        return oriented_arcs

print(projet.ConnexSets(liste_arcs))
print(projet.OrientConnexSets(train,liste_arcs,'target'))

#####
#Q 2.4 : Quelle classifieur préférez-vous en théorie entre APrioriClassifier , ML2DClassifier et MAP2DClassifier ? Quels résultats vous semble-les plus intéressants ?
#####
#Je préfère le classificateur MAP2DClassifier, car il me semble le plus complexe parmi les trois, étant donné 
# qu'il prend en compte à la fois les probabilités conditionnelles et a priori. 
# Les informations utilisées par ce classificateur sont plus complètes que celles utilisées par 
# le APrioriClassifier, qui ne considère que la probabilité a priori. En le comparant au ML2DClassifier, 
# ils me semblent trop similaires.
#####

#####
#Q 3.3.a : Preuve de l'indépendance conditionnelle
#####
#Si deux variables X et Y sont indépendantes conditionnellement à une troisième variable Z, cela signifie que, une fois que Z est connu, 
# la connaissance de X ne donne aucune information supplémentaire sur Y, et vice versa.
#Mathématiquement, l'indépendance conditionnelle de X et Y sachant Z est définie comme suit :
#P(X,Y∣Z)=P(X∣Z)P(Y∣Z)
#C'est-à-dire que la distribution jointe des variables X et Y, conditionnée sur Z, est égale au produit des distributions conditionnelles de 
#X et Y conditionnées sur Z.
#Par définition, X et Y sont indépendants conditionnellement à Z si, et seulement si, P(X,Y∣Z)=P(X∣Z)P(Y∣Z).
#Lois de probabilité conditionnelle :
#La probabilité conditionnelle peut être décomposée selon les règles de probabilité :
#P(X,Y∣Z)= P(Z)P(X,Y,Z), P(X∣Z)= P(Z)P(X,Z) et P(Y∣Z)= P(Z)P(Y,Z)
#Indépendance conditionnelle dans la loi jointe :
#Si X et Y sont indépendants conditionnellement à Z, la loi jointe
#P(X,Y∣Z) peut être factorisée comme suit : P(X,Y∣Z)= P(Z)P(X,Z)P(Y,Z)=P(X∣Z)P(Y∣Z)
#Cela montre que, sous condition de Z, X et Y sont indépendants, ce qui prouve l'indépendance conditionnelle.
#####

#####
#Q 3.3.b : Complexité en indépendance partielle
#####
#On a deux cas:
###1. Sans indépendance conditionnelle :
#Nous avons trois variables X, Y, et Z qui prennent respectivement kX, kY, et kZ valeurs distinctes.
#La taille mémoire nécessaire pour représenter cette distribution sans utiliser l'indépendance conditionnelle est :
#MEM = kX * kY *kZ * 8octet
###2.  Avec indépendance conditionnelle :
#SiX et Y sont indépendants conditionnellement à Z, la distribution jointe peut être factorisée comme suit :
#P(X,Y,Z)=P(X∣Z)P(Y∣Z)P(Z)
#Cela permet de représenter la distribution avec une table de probabilité plus petite, où :
#La taille de la table P(X∣Z) sera kX * kZ (le produit des valeurs possibles de X et Z),
#La taille de la table P(Y∣Z) sera kY *kZ *,
#La taille de la table P(Z) sera kZ.
#La taille mémoire nécessaire pour représenter cette distribution avec indépendance conditionnelle est donc :
#MEM = (kX * kZ * + kY * kZ +kZ)×8octets
#####

#####
#Q 4.2 : Naive Bayes
####
#Décomposition de la vraisemblance 
#P(attr1, attr2, ... | target) = P(attr1|target) * P(attr2|attr1, target) * P(attr3| attr2, atttr1, target) *.... 
#Décomposition de la distribution a posteriori
#P(taregt|attr1,attr2,...) = [P(attr1, attr2, ... | target) * P(target)] / P(attr1, attr2, ...)
#####

#####
#Q 6.1 : Où se trouve à votre avis le point idéal ? Comment pourriez-vous proposer de comparer les différents classifieurs dans cette représentation graphique?
#####
# A mon avis,  le point idéal se trouve dans le coin supérieur droit du graphique, où la précision (precision) est maximale (égale à 1) et le rappel (recall) est 
# également maximal (égal à 1). Cela signifie que le classifieur fait une prédiction parfaite, en classifiant correctement tous les exemples positifs et négatifs 
# sans aucun faux positif ni faux négatif.
#####

#####
#Q 6.3 : Qu'en concluez vous ?
#####
#Les différents points sur le graphique montrent que certains classificateurs (par exemple ceux proches du coin supérieur droit) obtiennent de meilleures performances en termes de précision et de rappel.
#Les classificateurs situés plus bas ou à gauche, MLNaiveBayesClassifier, MAPNaiveBayesClassifier, ReducedMAPNaiveBayesClassifier, ReducedMLNaiveBayesClassifier, ont des performances moins bonnes, soit parce qu’ils ont une précision ou un rappel inférieur.
#Le classificateur ML2DClassifier semble se démarquer, car il combine une précision et un rappel proches de 1, ce qui en fait un choix potentiellement idéal pour un compromis entre ces deux métriques.
#####

