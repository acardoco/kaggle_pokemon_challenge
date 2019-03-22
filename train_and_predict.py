import numpy as np
import pandas as pd

from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn import metrics

n_test= 49999
fichero = 'datos.txt'
tests = 'test.txt'
resultados_finales = 'resultados_finales/test.csv'
path_dir = 'pokemon-challenge-mlh/'
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------DATASET-----------------------------------------------------------
def get_data():

    #-------pokemon.csv
    df_pokemon = pd.read_csv(path_dir + 'pokemon.csv')
    df_pokemon = df_pokemon.fillna({'Name': 'Ninguno', 'Type 1': 'Ninguno', 'Type 2': 'Ninguno'})
    #df_pokemon = df_pokemon.dropna()
    #cambiando nombre de variable
    df_pokemon = df_pokemon.rename(index=str, columns={"#": "id_pokemon"})
    # encoding
    df_pokemon['Legendary'] = np.where(df_pokemon['Legendary'] == True, 1, 0)
    # encoding name, type1 y type2
    valores_type1 = df_pokemon['Type 1'].values
    valores_type2 = df_pokemon['Type 2'].values
    valores_name = df_pokemon['Name'].values

    #print(df_pokemon.isna().sum())

    le1 = preprocessing.LabelEncoder()
    le2 = preprocessing.LabelEncoder()
    lename = preprocessing.LabelEncoder()
    encoding1 = le1.fit_transform(valores_type1)
    encoding2 = le2.fit_transform(valores_type2)
    encodingName = lename.fit_transform(valores_name)

    # asignando
    df_pokemon['Type 1'] = encoding1
    df_pokemon['Type 2'] = encoding2
    df_pokemon['Name'] = encodingName

    print(df_pokemon.head())
    #-------battles.csv
    df_battles = pd.read_csv(path_dir + 'battles.csv')
    # quitamos el numero de batalla
    df_battles = df_battles[['First_pokemon','Second_pokemon', 'Winner']]
    print(df_battles.head())
    #-------test.csv
    df_test = pd.read_csv(path_dir + 'test.csv')

    return df_pokemon, df_battles, df_test


# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def juntar_csvs():
    df_pokemon, df_battles, df_test = get_data()

    lista = []
    i = 0
    for batalla in df_battles.values:
        for pokemon in df_pokemon.values:
            if batalla[0] == pokemon[0]:
                fila1 = pokemon[1:]
            if batalla[1] == pokemon[0]:
                fila2 = pokemon[1:]

        ids = np.array([batalla[0], batalla[1]])
        filas = np.concatenate((fila1, fila2), axis=0)
        primera_juntanza = np.concatenate((ids, filas), axis=0)
        lista_final = np.append(primera_juntanza, batalla[2])
        lista.append(lista_final)

        i += 1
        if i % 100 == 0:
            print(i)

    lista = np.asarray(lista)
    print(lista.shape)
    # guardo el fichero
    np.savetxt(fichero, lista)

    return lista
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def preparar_test():
    df_pokemon, df_battles, df_test = get_data()

    lista = []
    i = 0
    for batalla in df_test.values:
        for pokemon in df_pokemon.values:
            if batalla[1] == pokemon[0]:
                fila1 = pokemon[1:]
            if batalla[2] == pokemon[0]:
                fila2 = pokemon[1:]

        ids = np.array([batalla[1], batalla[2]])
        filas = np.concatenate((fila1, fila2), axis=0)
        lista_final = np.concatenate((ids, filas), axis=0)
        #lista_final = np.append(primera_juntanza, batalla[2])
        lista.append(lista_final)

        i += 1
        if i % 100 == 0:
            print(i)

    lista = np.asarray(lista)
    print(lista.shape)
    # guardo el fichero
    np.savetxt(tests, lista)
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------MODELO EMPLEADOS--------------------------------------------------
def random_forest(train_x, train_y, test_x, test_y):
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(train_x, train_y)

    y_pred=clf.predict(test_x)
    print("Accuracy random forest:",metrics.accuracy_score(test_y, y_pred))

    return clf
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def MLP(train_x, train_y, test_x, test_y):
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
    clf.fit(train_x, train_y)

    y_pred=clf.predict(test_x)
    print("Accuracy MLP:",metrics.accuracy_score(test_y, y_pred))

    return clf
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def SVM(train_x, train_y, test_x, test_y):
    clf = svm.SVC(gamma='scala')
    clf.fit(train_x, train_y)

    y_pred=clf.predict(test_x)
    print("Accuracy SVM:",metrics.accuracy_score(test_y, y_pred))

    return clf
# ----------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------
# ----------------------------------------------------RESULTADOS--------------------------------------------------
def solo_battles():
    df_pokemon, df_battles, df_test = get_data()
    # vamos a juntar ambos ficheros en uno para entrenar conjuntamente
    df_a_entrenar = df_battles[['First_pokemon','Second_pokemon', 'Winner']]
    X = df_a_entrenar[['First_pokemon','Second_pokemon']].values
    y = df_a_entrenar['Winner'].values
    train_x,train_y = X[:n_test], y[:n_test]
    test_x, test_y = X[n_test:], y[n_test:]

    rf = random_forest(train_x, train_y, test_x, test_y)
    mlp = MLP(train_x, train_y, test_x, test_y)
    #svm = SVM(train_x, train_y, test_x, test_y)
# ----------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------
def agrupados():

    lista = np.loadtxt(fichero)

    print(lista.shape)

    X = lista[:, :-1]
    y = lista[:, -1]

    train_x,train_y = X[:n_test], y[:n_test]
    test_x, test_y = X[n_test:], y[n_test:]

    rf = random_forest(train_x, train_y, test_x, test_y)
    mlp = MLP(train_x, train_y, test_x, test_y)
    #svm = SVM(train_x, train_y, test_x, test_y)

    return rf
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# ------------------------------------------guardar datos finales-------------------------------------------------------
def resultado_final():

    lista = np.loadtxt(tests)
    print(lista.shape)

    clf = agrupados()
    y_pred = clf.predict(lista)
    y_pred = y_pred.astype(int)

    df_test = pd.read_csv(path_dir + 'test.csv')
    df_test['Winner'] = y_pred
    df_test.to_csv(resultados_finales, index=False)

# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# solo_battles()
#juntar_csvs()
#preparar_test()
#agrupados()
resultado_final()
