import random as rn
import numpy as np

#from numpy.random import choice as np_choice

class AntColony(object):

    def __init__(self, distances, n_ants, n_best, n_iterations, evaporation, alpha=1, beta=1):
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
        self.distances  = distances
        self.pheromone = np.ones(self.distances.shape) / len(distances)
        self.all_inds = range(len(distances))
        self.n_ants = n_ants
        self.n_best = n_best
        self.n_iterations = n_iterations
        self.alpha = alpha
        self.beta = beta
        self.decay = 1 - evaporation

    def run(self):
        shortest_path = None
        all_time_shortest_path = ("placeholder", np.inf) # melhor elemento encontrado ate ao momento
        for i in range(self.n_iterations):
            all_paths = self.gen_all_paths_random()
            self.spread_pheronome(all_paths, self.n_best, shortest_path=shortest_path)
            shortest_path = min(all_paths, key=lambda x: x[1])
            print (i,shortest_path)
            if shortest_path[1] < all_time_shortest_path[1]:
                all_time_shortest_path = shortest_path            
            self.pheromone *= self.decay            
        return all_time_shortest_path

    def spread_pheronome(self, all_paths, n_best, shortest_path):
        sorted_paths = sorted(all_paths, key=lambda x: x[1])
        for path, dist in sorted_paths[:n_best]:
            for move in path:
                self.pheromone[move] += 1.0 / self.distances[move]


    #devolve o comprimento de um caminho
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

    #generate a path randomly with all the cities [(1,2), (2,4), (4,8), (8,1)]
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
        path.append((prev, start)) # going back to where we started, add connection to the first node    
        return path

    # determina probabilisticamente a proxima cidade a visitar
    def pick_city(self, pheromone, dist, visited):
        pheromone = np.copy(pheromone)
        pheromone[list(visited)] = 0

        row = pheromone ** self.alpha * (( 1.0 / dist) ** self.beta)

        norm_row = row / row.sum()
        #move = np_choice(self.all_inds, 1, p=norm_row)[0]
        move = np.random.choice(self.all_inds, 1, p=norm_row)[0] # [0] - devolve o indice do elemento, 1 - 1 elemento, p - probablidade por elemento
        return move


distancias = np.genfromtxt("distancias.txt", dtype='i', delimiter='\t') #usecols=(1,2,3,4,5,6,7,8,9,10))
cities = np.genfromtxt("cidades.txt", dtype=None, delimiter='\n', encoding='utf-8')

print(distancias)

pop = AntColony(distancias,n_ants=100, n_best=10, n_iterations=100, evaporation=0.05)
best = pop.run()
print('Best in all the iterations:')
print(best)
print("Total distance traveled: %d km" % best[1])
l=[]
for k in best[0]:
    l.append(cities[k[0]])
    #print ((cities[k[0]],cities[k[1]]))
print(l)

x = lambda x: ' '+ str(x) if x <= 9 else str(x) 
for k in range(len(cities)):
    print( x(k) + ' -- ' + l[k])


