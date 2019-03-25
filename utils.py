import numpy as np
import pandas as pd

# via https://www.kaggle.com/terminus7/pokemon-challenge
def calculate_effectiveness(data):
    '''
        this function creates a new column of each pokemon's effectiveness against it's enemy.
        every effectiveness starts with 1, if an effective type is found on enemy's type, effectiveness * 2
        if not very effective is found on enemy's type, effectiveness / 2
        if not effective is found on enemy's type, effectiveness * 0

        This function creates 4 new columns
            1. P1_type1, pokemon 1 first type effectiveness against the enemy's type
            2. P1_type2, pokemon 1 second type effectiveness against the enemy's type
            3. P2_type1, pokemon 2 first type effectiveness against the enemy's type
            4. P2_type2, pokemon 2 second type effectiveness against the enemy's type
    '''

    very_effective_dict = {'Normal': [],
                           'Fighting': ['Normal', 'Rock', 'Steel', 'Ice', 'Dark'],
                           'Flying': ['Fighting', 'Bug', 'Grass'],
                           'Poison': ['Grass', 'Fairy'],
                           'Ground': ['Poison', 'Rock', 'Steel', 'Fire', 'Electric'],
                           'Rock': ['Flying', 'Bug', 'Fire', 'Ice'],
                           'Bug': ['Grass', 'Psychic', 'Dark'],
                           'Ghost': ['Ghost', 'Psychic'],
                           'Steel': ['Rock', 'Ice', 'Fairy'],
                           'Fire': ['Bug', 'Steel', 'Grass', 'Ice'],
                           'Water': ['Ground', 'Rock', 'Fire'],
                           'Grass': ['Ground', 'Rock', 'Water'],
                           'Electric': ['Flying', 'Water'],
                           'Psychic': ['Fighting', 'Poison'],
                           'Ice': ['Flying', 'Ground', 'Grass', 'Dragon'],
                           'Dragon': ['Dragon'],
                           'Dark': ['Ghost', 'Psychic'],
                           'Fairy': ['Fighting', 'Dragon', 'Dark'],
                           'None': []}

    not_very_effective_dict = {'Normal': ['Rock', 'Steel'],
                               'Fighting': ['Flying', 'Poison', 'Bug', 'Psychic', 'Fairy'],
                               'Flying': ['Rock', 'Steel', 'Electric'],
                               'Poison': ['Poison', 'Rock', 'Ground', 'Ghost'],
                               'Ground': ['Bug', 'Grass'],
                               'Rock': ['Fighting', 'Ground', 'Steel'],
                               'Bug': ['Fighting', 'Flying', 'Poison', 'Ghost', 'Steel', 'Fire', 'Fairy'],
                               'Ghost': ['Dark'],
                               'Steel': ['Steel', 'Fire', 'Water', 'Electric'],
                               'Fire': ['Rock', 'Fire', 'Water', 'Dragon'],
                               'Water': ['Water', 'Grass', 'Dragon'],
                               'Grass': ['Flying', 'Poison', 'Bug', 'Steel', 'Fire', 'Grass', 'Dragon'],
                               'Electric': ['Grass', 'Electric', 'Dragon'],
                               'Psychic': ['Steel', 'Psychic'],
                               'Ice': ['Steel', 'Fire', 'Water', 'Psychic'],
                               'Dragon': ['Steel'],
                               'Dark': ['Fighting', 'Dark', 'Fairy'],
                               'Fairy': ['Posion', 'Steel', 'Fire'],
                               'None': []}

    not_effective_dict = {'Normal': ['Ghost'],
                          'Fighting': ['Ghost'],
                          'Flying': [],
                          'Poison': ['Steel'],
                          'Ground': ['Flying'],
                          'Rock': [],
                          'Bug': [],
                          'Ghost': ['Normal'],
                          'Steel': [],
                          'Fire': [],
                          'Water': [],
                          'Grass': [],
                          'Electric': ['Ground'],
                          'Psychic': ['Dark'],
                          'Ice': [],
                          'Dragon': ['Fairy'],
                          'Dark': [],
                          'Fairy': [],
                          'None': []}

    p1_type1_list = []
    p1_type2_list = []
    p1_max = []
    p2_type1_list = []
    p2_type2_list = []
    p2_max = []

    for row in data.itertuples():
        nested_type = [[1, 1], [1, 1]]

        tipos_pok_1 = [row.tipo1_id1, row.tipo2_id1]
        tipos_pok_2 = [row.tipo1_id2, row.tipo2_id2]

        # manipulating values if found on dictionary
        for i in range(0, 2):
            for j in range(0, 2):
                if tipos_pok_2[j] in very_effective_dict.get(tipos_pok_1[i]):
                    nested_type[0][i] *= 2
                if tipos_pok_2[j] in not_very_effective_dict.get(tipos_pok_1[i]):
                    nested_type[0][i] /= 2
                if tipos_pok_2[j] in not_effective_dict.get(tipos_pok_1[i]):
                    nested_type[0][i] *= 0

                if tipos_pok_1[j] in very_effective_dict.get(tipos_pok_2[i]):
                    nested_type[1][i] *= 2
                if tipos_pok_1[j] in not_very_effective_dict.get(tipos_pok_2[i]):
                    nested_type[1][i] /= 2
                if tipos_pok_1[j] in not_effective_dict.get(tipos_pok_2[i]):
                    nested_type[1][i] *= 0

        p1_type1_list.append(nested_type[0][0])
        p1_type2_list.append(nested_type[0][1])
        p2_type1_list.append(nested_type[1][0])
        p2_type2_list.append(nested_type[1][1])
        p1_max.append(np.maximum(nested_type[0][0], nested_type[0][1]))
        p2_max.append(np.maximum(nested_type[1][0], nested_type[1][1]))

    data = data.assign(P1_type1=p1_type1_list, P1_type2=p1_type2_list,
                       P2_type1=p2_type1_list, P2_type2=p2_type2_list)
    #data = data.drop(['First_pokemon', 'Second_pokemon'], axis=1)

    return data

# con esto no aportamos mucho al modelo asi que queda aqui, pero descartada de aplicar
def get_winrate(df_pokemon, df_battles):

    batallas = df_battles.values
    pokemons = df_pokemon.values
    id_pokemons = pokemons[:, 0]

    won_battles = np.zeros((len(id_pokemons)))

    for idx, id in enumerate(id_pokemons):
        # first_pok, second_pok, winner [0 | 1]
        total_battles = batallas[(batallas[:,0] == id) | (batallas[:,1] == id)]
        izq_won = total_battles[(id == total_battles[:, 0]) &
                                (total_battles[:, 2] == 0)]
        der_won = total_battles[(id == total_battles[:, 1]) &
                                (total_battles[:, 2] == 1)]
        total_won = len(izq_won) + len(der_won)

        # y calculamos el winrate
        if len(total_battles) > 0 :
            winrate = total_won/len(total_battles)

        won_battles[idx] += winrate

    # asignamos
    df_pokemon['Winrate'] = won_battles

    return df_pokemon