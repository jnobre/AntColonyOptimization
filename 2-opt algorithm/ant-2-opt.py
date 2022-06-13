import random as rn
import numpy as np


# from numpy.random import choice as np_choice

class TwoOpt(object):

    def __init__(self, distances, n_iterations, improvement_threshold=0.001, alpha=1, beta=1):
        """
        Args:
            distances (2D numpy.array): Square matrix of distances. Diagonal is assumed to be np.inf.
            n_ants (int): Number of ants running per iteration
            n_best (int): Number of best ants who deposit pheromone
            n_iteration (int): Number of iterations
            decay (float): Rate it which pheromone decays. The pheromone value is multiplied by decay, so 0.95 will lead to decay, 0.5 to much faster decay.
            alpha (int or float): exponenet on pheromone, higher alpha gives pheromone more weight. Default=1
            beta (int or float): exponent on distance, higher beta give distance more weight. Default=1
        Example:
            ant_colony = AntColony(german_distances, 100, 20, 2000, 0.95, alpha=1, beta=2)
        """
        self.distances = distances
        self.pheromone = np.ones(self.distances.shape) / len(distances)
        self.all_inds = range(len(distances))
        self.n_iterations = n_iterations
        self.improvement_threshold = improvement_threshold
        self.alpha = alpha
        self.beta = beta

    def run(self):
        shortest_path = None
        all_time_shortest_path = ("placeholder", np.inf) # melhor elemento encontrado ate ao momento
        for i in range(self.n_iterations):
            path = self.gen_path_random(0)
            shortest_path = self.two_opt(path)
            print(i, shortest_path)
            if shortest_path[1] < all_time_shortest_path[1]:
                all_time_shortest_path = shortest_path
        return all_time_shortest_path

    def two_opt(self, route):
        improvement_factor = 1
        best_distance = self.path_dist(route)

        # Continua ate nao encontrar melhorias
        while improvement_factor > self.improvement_threshold:
            distance_to_beat = best_distance
            for first in range(1, len(route) - 2):
                for last in range(first + 1, len(route)):
                    # Troca a posicao de duas cidades
                    new_route = self.two_opt_swap(route, first, last)

                    # Verifica a distancia da nova rota
                    new_distance = self.path_dist(new_route)

                    if new_distance < best_distance:
                        route = new_route
                        best_distance = new_distance

            # Calcula o fator de melhoria
            improvement_factor = 1 - best_distance / distance_to_beat
        return (route, best_distance)

    # Troca a posicao de dois elementos
    def two_opt_swap(self, r, i, k):
        temp_array = r.copy()

        # Troca a posição das cidades
        temp = temp_array[i]
        temp_array[i] = temp_array[k]
        temp_array[k] = temp

        # Ajusta as rotas
        path = []
        size = len(r) - 1
        for i in range(size):
            path.append((temp_array[i][0], temp_array[i + 1][0]))
        path.append((temp_array[size][0], temp_array[0][0]))

        return path

    def spread_pheronome(self, all_paths, n_best, shortest_path):
        sorted_paths = sorted(all_paths, key=lambda x: x[1])
        for path, dist in sorted_paths[:n_best]:
            for move in path:
                self.pheromone[move] += 1.0 / self.distances[move]

    # devolve o comprimento de um caminho
    def path_dist(self, path):
        dist = 0
        for ele in path:
            dist += self.distances[ele]
        return dist

    # gera uma path por formiga, devolve uma lista: [[[(1,2), (2,4), (4,8), (8,1)],dist], [[(1,2), (2,4), (4,8), (8,1)],dist], [[(1,2), (2,4), (4,8), (8,1)],dist]]
    def gen_all_paths_random(self):
        all_paths = []
        for i in range(self.n_ants):
            path = self.gen_path_random(0)
            all_paths.append((path, self.path_dist(path)))
        return all_paths

    # generate a path randomly with all the cities [(1,2), (2,4), (4,8), (8,1)]
    def gen_path_random(self, start):
        path = []
        visited = set()
        visited.add(start)
        prev = start
        for i in range(len(self.distances) - 1):
            move = self.pick_city(self.pheromone[prev], self.distances[prev], visited)
            path.append((prev, move))
            prev = move
            visited.add(move)
        path.append((prev, start))  # going back to where we started, add connection to the first node
        return path

    # determina probabilisticamente a proxima cidade a visitar
    def pick_city(self, pheromone, dist, visited):
        pheromone = np.copy(pheromone)
        pheromone[list(visited)] = 0

        row = pheromone ** self.alpha * ((1.0 / dist) ** self.beta)

        norm_row = row / row.sum()
        # move = np_choice(self.all_inds, 1, p=norm_row)[0]
        move = np.random.choice(self.all_inds, 1, p=norm_row)[
            0]  # [0] - devolve o indice do elemento, 1 - 1 elemento, p - probablidade por elemento
        return move


distancias = np.genfromtxt("../distancias.txt", dtype='i', delimiter='\t')  # usecols=(1,2,3,4,5,6,7,8,9,10))
cities = np.genfromtxt("../cidades.txt", dtype=None, delimiter='\n', encoding='utf-8')

print(distancias)

pop = TwoOpt(distancias, n_iterations=500, improvement_threshold=0.0000000000001)
best = pop.run()
print('Best in all the iterations:')
print(best)
l = []
for k in best[0]:
    l.append(cities[k[0]])
    # print ((cities[k[0]],cities[k[1]]))
print(l)

x = lambda x: ' ' + str(x) if x <= 9 else str(x)
for k in range(len(cities)):
    print(x(k) + ' -- ' + cities[k])
