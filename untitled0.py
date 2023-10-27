import numpy as np
import warnings
from math import sqrt
from collections import Counter


def k_nearest_neighbors(data, predict, k=3):
    if len(data) >= k:
        warnings.warn('K is set to a value less than total voting groups!')
        
    distances = []
    for group in data:
        for features in data[group]:
            euclidean_distance = np.linalg.norm(np.array(features)-np.array(predict))
            distances.append([euclidean_distance,group])
            

    lucas = sorted(distances)[:3]
    votes = [i[1] for i in sorted(distances)[:k]]  #ele so esta pegando a lista gerada por sorted(distances)[:k]] e pegando o valor da classe 'r' ou 'k'
    
    
    vote_result = Counter(votes).most_common(1)[0][1]  # [0][1] pra acessar o segundo valor da tupla e [0]  para retornar a tupla, lembrando que Counter(votes).most_common(1) resulta numa lista aninhada ou lista dentro de uma lista
    #por isso precisamos disso
    return vote_result 

dataset = {'k':[[1,2],[2,3],[3,1]], 'r':[[6,5],[7,7],[8,6]]}
new_features = [5,7]

result =  k_nearest_neighbors(dataset, new_features, k=3)

print(result)



