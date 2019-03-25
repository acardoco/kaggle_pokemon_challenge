import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder

import utils

n_test= 49999
fichero = 'datos.csv'
tests = 'entrega_para_predecir.csv'
resultados_finales = 'resultados_finales/test.csv'
sample = 'resultados_finales/sampleSubmission.csv'
path_dir = 'pokemon-challenge-mlh/'
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------DATASET-----------------------------------------------------------
def get_data():

    #-------pokemon.csv
    df_pokemon = pd.read_csv(path_dir + 'pokemon.csv')
    df_pokemon = df_pokemon.fillna({'Name': 'None', 'Type 1': 'None', 'Type 2': 'None'})
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

    # rapido -> 1, Lento -> 0
    sum_speeds = np.sum(df_pokemon['Speed'].values)
    total_speeds = len(df_pokemon['Speed'])
    media_speeds = sum_speeds / total_speeds
    df_pokemon['Rapidez'] = np.where(df_pokemon['Speed'] > media_speeds, 1, 0)

    #-------battles.csv
    df_battles = pd.read_csv(path_dir + 'battles.csv')
    # quitamos el numero de batalla
    df_battles = df_battles[['First_pokemon','Second_pokemon', 'Winner']]
    print(df_battles.columns)

    #winrates
    #df_pokemon = utils.get_winrate(df_pokemon, df_battles)
    print(df_pokemon.head())

    #-------test.csv
    df_test = pd.read_csv(path_dir + 'test.csv')

    return df_pokemon, df_battles, df_test, le1, le2, lename


# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def juntar_csvs():
    df_pokemon, df_battles, df_test, le1, le2, lename = get_data()

    #vectorizacion
    pokemon_values = df_pokemon.values #(800, cols)
    battles_values = df_battles.values #(50000, 3)
    
    ids_pokemon = pokemon_values[:,0]
    # obtenemos valores unicos y los indices inversos para luego reconstruir el array original
    ids_pok1, inv1 = np.unique(battles_values[:, 0], return_inverse=True)
    ids_pok2, inv2 = np.unique(battles_values[:, 1], return_inverse=True)
    resultados_batallas = battles_values[:, 2]

    # buscamos donde estan las caracteristicas de cada pokemon en las batallas
    indices1 = np.intersect1d(ids_pok1, ids_pokemon, return_indices=True)
    indices2 = np.intersect1d(ids_pok2, ids_pokemon, return_indices=True)

    # asignamos las caracteristicas
    vals_pok1 = pokemon_values[indices1[2], 1:]
    vals_pok2 = pokemon_values[indices2[2], 1:]

    # pokemons sin batallas
    sin_battles = pokemon_values[
        np.where(
            np.logical_not(
                np.isin(ids_pokemon, ids_pok1)))]
    # 16 en total
    print('Pokemons que no han peleado:', len(sin_battles))

    # y reconstruimos el array original
    lon_values = len(battles_values)
    # (50000, 11) cada uno
    pok1 = vals_pok1[inv1]
    pok2 = vals_pok2[inv2]
    #columnas = pok2.shape[1] * 2
    columnas = pok2.shape[1] + 3 #nombre2,tipo1_id2,tipo2_id2, el mas rapido
    print(pok2.shape)

    # aplicamos diff
    pok_final = np.ones((lon_values, columnas))
    pok_final[:, :3] = pok1[:, :3]#nombre1,tipo1_id1,tipo2_id1
    pok_final[:, 3:6] = pok2[:, :3]#nombre2,tipo2_id2,tipo2_id2
    pok_final[:, 6:] = pok1[:, 3:] - pok2[:, 3:]
    # el mas rapido
    #pok_final[:, -1] = np.where(pok1[:, -4] > pok2[:, -4], battles_values[:, 0], battles_values[:, 1])

    # aqui juntamos el resto para crear el dataset con el que entrenar
    #juntar_carac = np.concatenate((pok1, pok2), axis=1)
    juntar_carac = pok_final
    caracteristicas_y_resultados = np.ones((lon_values, columnas + 1)) # (50000, 15)
    caracteristicas_y_resultados[:,:-1] = juntar_carac
    caracteristicas_y_resultados[:,-1] = resultados_batallas

    # ids contrincante 1, ids contrincante 2 y el que golpea primero (añadido)
    valores = np.array((battles_values[:, 0], battles_values[:, 1], battles_values[:, 0])) #(3, 50000)
    valores = valores.T #(50000, 3)

    lista = np.concatenate((valores, caracteristicas_y_resultados), axis=1)
    lista = lista.astype(int)

    # guardo el fichero
    df_lista = pd.DataFrame(lista, columns=['First_pokemon', 'Second_pokemon', 'id_primer_ataq',
                                            'nombre1', 'tipo1_id1', 'tipo2_id1',
                                            'nombre2', 'tipo1_id2', 'tipo2_id2',
                                            'diff_HP','diff_Attack','diff_Defense','diff_Sp. Atk','diff_Sp. Def','diff_Speed',
                                            'diff_Generation', 'diff_Legendary',
                                            'diff_Rapidez',
                                            'Winner'])

    # efectividad de las habilidades
    # primero pasamos a las antiguas labels
    df_lista['tipo1_id1'] = le1.inverse_transform(df_lista['tipo1_id1'])
    df_lista['tipo2_id1'] = le2.inverse_transform(df_lista['tipo2_id1'])
    df_lista['tipo1_id2'] = le1.inverse_transform(df_lista['tipo1_id2'])
    df_lista['tipo2_id2'] = le2.inverse_transform(df_lista['tipo2_id2'])
    df_lista['nombre1'] = lename.inverse_transform(df_lista['nombre1'])
    df_lista['nombre2'] = lename.inverse_transform(df_lista['nombre2'])

    # y luego aplicamos los valores
    df_lista = utils.calculate_effectiveness(df_lista)

    # reordenamos para colocar la columnas Winner al final
    winners = df_lista['Winner'].values
    df_lista = df_lista.drop(['Winner'], axis=1)
    df_lista['Winner'] = winners

    #y volvemos a aplicar los encodings
    df_lista['tipo1_id1'] = le1.fit_transform(df_lista['tipo1_id1'])
    df_lista['tipo2_id1'] = le2.fit_transform(df_lista['tipo2_id1'])
    df_lista['tipo1_id2'] = le1.fit_transform(df_lista['tipo1_id2'])
    df_lista['tipo2_id2'] = le2.fit_transform(df_lista['tipo2_id2'])
    df_lista['nombre1'] = lename.fit_transform(df_lista['nombre1'])
    df_lista['nombre2'] = lename.fit_transform(df_lista['nombre2'])

    # elimino carac que aportan menos --> no aporta
    #df_lista = df_lista.drop(['diff_Generation', 'diff_Legendary'], axis=1)

    df_lista.to_csv(fichero, index=False)
    #np.savetxt(fichero, lista)

    return lista
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def preparar_test():
    df_pokemon, df_battles, df_test, le1, le2, lename = get_data()

    # vectorizacion
    pokemon_values = df_pokemon.values  # (800, 12)
    tests_values = df_test.values  # (10000, 3)

    ids_pokemon = pokemon_values[:, 0]
    # obtenemos valores unicos y los indices inversos para luego reconstruir el array original
    ids_pok1, inv1 = np.unique(tests_values[:, 1], return_inverse=True)
    ids_pok2, inv2 = np.unique(tests_values[:, 2], return_inverse=True)

    # buscamos donde estan las caracteristicas de cada pokemon en las batallas
    indices1 = np.intersect1d(ids_pok1, ids_pokemon, return_indices=True)
    indices2 = np.intersect1d(ids_pok2, ids_pokemon, return_indices=True)

    # asignamos las caracteristicas
    vals_pok1 = pokemon_values[indices1[2], 1:]
    vals_pok2 = pokemon_values[indices2[2], 1:]

    # pokemons sin batallas
    sin_battles = pokemon_values[
        np.where(
            np.logical_not(
                np.isin(ids_pokemon, ids_pok1)))]
    # 16 en total
    print('Pokemons que no han peleado en test:', len(sin_battles))

    # y reconstruimos el array original
    lon_values = len(tests_values)
    # (10000, 11) cada uno
    pok1 = vals_pok1[inv1]
    pok2 = vals_pok2[inv2]
    columnas = pok2.shape[1] + 3  # nombre2,tipo1_id2,tipo2_id2, Mas_Winrate

    # aplicamos diff
    pok_final = np.ones((lon_values, columnas))
    pok_final[:, :3] = pok1[:, :3]
    pok_final[:, 3:6] = pok2[:, :3]
    pok_final[:, 6:] = pok1[:, 3:] - pok2[:, 3:]
    # winrate
    #pok_final[:, -2] = np.where(pok1[:, -1] > pok2[:, -1], tests_values[:, 0], tests_values[:, 1])
    # el mas rapido
    #pok_final[:, -1] = np.where(pok1[:, -2] > pok2[:, -2], tests_values[:, 0], tests_values[:, 1])

    # aqui juntamos el resto para crear el dataset con el que entrenar
    # juntar_carac = np.concatenate((pok1, pok2), axis=1)
    juntar_carac = pok_final

    # ids contrincante 1, ids contrincante 2 y el que golpea primero (añadido)
    valores = np.array((tests_values[:, 1], tests_values[:, 2], tests_values[:, 1]))  # (3, 10000)
    valores = valores.T  # (10000, 3)

    lista = np.concatenate((valores, juntar_carac), axis=1)
    lista = lista.astype(int)
    print(lista.shape)
    # guardo el fichero
    df_lista = pd.DataFrame(lista, columns=['First_pokemon', 'Second_pokemon', 'id_primer_ataq',
                                            'nombre1', 'tipo1_id1', 'tipo2_id1',
                                            'nombre2', 'tipo1_id2', 'tipo2_id2',
                                            'HP','Attack','Defense','Sp. Atk','Sp. Def','Speed',
                                            'Generation', 'Legendary',
                                            'Rapidez'
                                            ])

    # efectividad de las habilidades
    # primero pasamos a las antiguas labels
    df_lista['tipo1_id1'] = le1.inverse_transform(df_lista['tipo1_id1'])
    df_lista['tipo2_id1'] = le2.inverse_transform(df_lista['tipo2_id1'])
    df_lista['tipo1_id2'] = le1.inverse_transform(df_lista['tipo1_id2'])
    df_lista['tipo2_id2'] = le2.inverse_transform(df_lista['tipo2_id2'])
    df_lista['nombre1'] = lename.inverse_transform(df_lista['nombre1'])
    df_lista['nombre2'] = lename.inverse_transform(df_lista['nombre2'])

    # y luego aplicamos los valores
    df_lista = utils.calculate_effectiveness(df_lista)

    # y volvemos a aplicar los encodings
    df_lista['tipo1_id1'] = le1.fit_transform(df_lista['tipo1_id1'])
    df_lista['tipo2_id1'] = le2.fit_transform(df_lista['tipo2_id1'])
    df_lista['tipo1_id2'] = le1.fit_transform(df_lista['tipo1_id2'])
    df_lista['tipo2_id2'] = le2.fit_transform(df_lista['tipo2_id2'])
    df_lista['nombre1'] = lename.fit_transform(df_lista['nombre1'])
    df_lista['nombre2'] = lename.fit_transform(df_lista['nombre2'])

    # elimino carac que aportan menos --> no aporta
    #df_lista = df_lista.drop(['Generation', 'Legendary'], axis=1)

    df_lista.to_csv(tests, index=False)

# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------MODELO EMPLEADOS--------------------------------------------------
def random_forest(train_x, train_y, test_x, test_y):

    clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.08, subsample=0.75,max_depth=7)
    clf.fit(train_x, train_y)

    y_pred=clf.predict(test_x)
    print(clf.feature_importances_)
    print("Accuracy random forest:",metrics.accuracy_score(test_y, y_pred))

    return clf
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def Voting(train_x, train_y, test_x, test_y):

    clf1 = RandomForestClassifier(n_estimators=150)
    clf2 = RandomForestClassifier(n_estimators=200)
    clf3 = RandomForestClassifier(n_estimators=175)
    '''clf3 = MLPClassifier(solver='lbfgs', alpha=1e-5
                         , hidden_layer_sizes=(10, 10), random_state=1, activation='logistic')'''

    eclf = VotingClassifier(estimators=[('rf1', clf1), ('rf2', clf2),('rf3', clf3)], voting='hard')

    clf1 = clf1.fit(train_x, train_y)
    clf2 = clf2.fit(train_x, train_y)
    clf3 = clf3.fit(train_x, train_y)
    eclf = eclf.fit(train_x, train_y)

    y_pred1 = clf1.predict(test_x)
    y_pred2 = clf2.predict(test_x)
    y_pred3 = clf3.predict(test_x)
    e_pred = eclf.predict(test_x)

    print("Accuracy RandomForestClassifier 150:", metrics.accuracy_score(test_y, y_pred1))
    print("Accuracy RandomForestClassifier 200:", metrics.accuracy_score(test_y, y_pred2))
    print("Accuracy RandomForestClassifier 175:", metrics.accuracy_score(test_y, y_pred3))
    print("Accuracy VotingClassifier:", metrics.accuracy_score(test_y, e_pred))

    return eclf
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------
# ----------------------------------------------------RESULTADOS--------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------
def agrupados():

    lista = pd.read_csv(fichero).values

    print(lista.shape)

    X = lista[:, :-1]
    y = lista[:, -1]

    train_x,train_y = X[:n_test], y[:n_test]
    test_x, test_y = X[n_test:], y[n_test:]

    rf = random_forest(train_x, train_y, test_x, test_y)
    #mlp = MLP(train_x, train_y, test_x, test_y)
    #svm = SVM(train_x, train_y, test_x, test_y)

    return rf
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# ------------------------------------------guardar datos finales-------------------------------------------------------
def resultado_final():

    lista = pd.read_csv(tests).values
    print(lista.shape)

    clf = agrupados()
    y_pred = clf.predict(lista)
    y_pred = y_pred.astype(int)

    df_test = pd.read_csv(path_dir + 'test.csv')
    df_test['Winner'] = y_pred
    df_test.to_csv(resultados_finales, index=False)

    df_sample = df_test[['battle_number', 'Winner']]
    df_sample.to_csv(sample, index=False)

# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# solo_battles()
juntar_csvs()
#preparar_test()
agrupados()
#resultado_final()