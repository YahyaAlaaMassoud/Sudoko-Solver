from genetic_evolution import GeneticEvolution

GA = GeneticEvolution(maximize = True, chromosome_len = 16)
scores, agents, sols = GA.Evolve(epochs = 10, population_size = 1000)