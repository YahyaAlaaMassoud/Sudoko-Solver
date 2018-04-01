from random import random, randint, shuffle
import numpy as np
import time
import glob, os, sys, getopt
import matplotlib.pyplot as plt
from individual import Individual

class GeneticEvolution():
    def __init__(self, max_age = 20, selection_rate = 0.03, mutate_rate = 0.08, elitism_rate = 0.2, crossover_rate = 0.8, crossover_operator = "random", mutation_operator = "random", maximize = True, chromosome_len = None):
        self.mutate_rate = mutate_rate
        self.elitism_rate = elitism_rate
        self.crossover_operator = crossover_operator
        self.mutation_operator = mutation_operator
        self.maximize = maximize
        self.chromosome_len = chromosome_len
        self.selection_rate = selection_rate
        self.crossover_rate = crossover_rate
        self.max_age = max_age
        
    def generate_init_population(self, population_size):
        return [Individual(self.chromosome_len, np.int(np.sqrt(self.chromosome_len))) for x in range(population_size)]
    
    def calculate_fitness(self, agent):
        return agent.get_fitness()
    
    def sort_agents_by_fitness(self, agents):
        return sorted(agents, reverse = self.maximize)
    
    def tournament_selection(self, agents, k):
        tournament_agents = []

        for i in range(k):
            index = randint(0, len(agents) - 1)
            tournament_agents.append(agents[index])
            
        tournament_agents = set(tournament_agents)
        tournament_agents = self.sort_agents_by_fitness(tournament_agents)

        return tournament_agents[0][0]
    
    def generate_new_population(self, population_size, agents = []):
        
        coupled = {}
        children = []
        selected_top_agents = []
        top_agents = agents[:int(len(agents) * self.elitism_rate)]
        for agent in top_agents:
            if agent[0].get_age() <= self.max_age:
                agent[0].increment_age()
                selected_top_agents.append(agent[0])
        
        while len(children) < (population_size - len(selected_top_agents)):
            r = int(len(agents) * self.selection_rate)
            male, female = self.tournament_selection(agents, r), self.tournament_selection(agents, r)
            if random() > self.crossover_rate:#coupled.get((male, female)) == None and male != female:
                new_offspring = self.crossover(male, female)
                coupled[(male, female)] = True
                coupled[(female, male)] = True
                if self.mutate_rate > random():
                    new_offspring = self.mutate(new_offspring)
                children.append(new_offspring)
#            print(str(len(children)) + ' ' + str(population_size))
#            elif coupled.get((male, female)) != None:
#                print('already coupled')
#            elif male == female:
#                print('same')
        
#        print('pop size: ' + str(population_size))
#        print('len of selected agents: ' + str(len(top_agents)))
#        print('len of new agents: ' + str(len(children)))
#        print()
        
        new_individuals = []
        for child in children:
            individual = Individual(self.chromosome_len, np.int(np.sqrt(self.chromosome_len)))
            individual.set_data(child)
            individual.increment_age()
            new_individuals.append(individual)

        return selected_top_agents + new_individuals
    
    def crossover(self, male, female):
        male_dna = male.get_data()
        female_dna = female.get_data()
        new_dna = []
        if self.crossover_operator == "random":
            r = randint(0, 1)
            if r == 0:
                for i in range(len(male_dna)):
                    if random() > 0.5:
                        new_dna.append(male_dna[i])
                    else:
                        new_dna.append(female_dna[i])
            else:
                i, j = 0, 0
                while i >= j:
                    i = randint(0, len(male_dna) - 1)
                    j = randint(0, len(female_dna) - 1)
                new_dna = male_dna[:i] + female_dna[i:j] + male_dna[j:]
                
        elif self.crossover_operator == "uniform":
            for i in range(len(male_dna)):
                if random() > 0.5:
                    new_dna.append(male_dna[i])
                else:
                    new_dna.append(female_dna[i])
                    
        elif self.crossover_operator == "two-point":
            i, j = 0, 0
            while i >= j:
                i = randint(0, len(male_dna) - 1)
                j = randint(0, len(male_dna) - 1)
            new_dna = male_dna[:i] + female_dna[i:j] + male_dna[j:]
            
        return new_dna
    
    def mutate(self, agent):
#        print('mutate')
        
        if self.mutation_operator == "random":
            r = randint(0, 3)
            if r == 0:
                agent = self.mutate_resetting(agent)
            elif r == 1:
                agent = self.mutate_swap(agent)
            elif r == 2:
                agent = self.mutate_scramble(agent)
            else:
                agent = self.mutate_inverse(agent)
        
        elif self.mutation_operator == "resetting":
            agent = self.mutate_resetting(agent)
            
        elif self.mutation_operator == "swap":
            agent = self.mutate_swap(agent)
        
        elif self.mutation_operator == "scramble":
            agent = self.mutate_scramble(agent)
            
        elif self.mutation_operator == "inverse":
            agent = self.mutate_inverse(agent)
            
#        print(bitstring.BitArray(bin = ''.join(map(str, X))).float)
        return agent
    
    def mutate_resetting(self, params):
        mutate_position = randint(0, len(params) - 1)
        params[mutate_position] = randint(1, np.int(np.sqrt(self.chromosome_len)))
        return params
    
    def mutate_swap(self, params):
        i, j = 0, 0
        while i == j:
            i = randint(0, len(params) - 1)
            j = randint(0, len(params) - 1)
        params[i], params[j] = params[j], params[i]
        return params
    
    def mutate_scramble(self, params):
        i, j = 0, 0
        while i >= j:
            i = randint(0, len(params) - 1)
            j = randint(0, len(params) - 1)
        cpy = params[i:j]
        shuffle(cpy)
        params[i:j] = cpy
        return params
    
    def mutate_inverse(self, params):
        i, j = 0, 0
        while i >= j:
            i = randint(0, len(params) - 1)
            j = randint(0, len(params) - 1)
        cpy = params[i:j][::-1]
        params[i:j] = cpy
        return params
        
    def Evolve(self, epochs, population_size, old_agents = [], generate_population = True):
        
        if generate_population == True:
            agents = self.generate_init_population(population_size)
        else:
            agents = old_agents
        
        all_scores = []
        
        for epoch in range(epochs):
            agents_scores = [(x, x.get_fitness()) for x in agents]
            agents_sorted = self.sort_agents_by_fitness(agents_scores)
            all_scores.append(agents_sorted)
#            if epoch % 20 == 0:
            print('current fittest: ' + str(agents_sorted[0][1]) + ' in epoch: ' + str(epoch) + ' with age: ' + str(agents_sorted[0][0].get_age()))
            print(str(np.array(agents_sorted[0][0].get_data()).reshape(np.int(np.sqrt(self.chromosome_len)), np.int(np.sqrt(self.chromosome_len)))))
            print()
            if agents_sorted[0][1] == 1.0:
                print("WE FOUND A SOLUTION")
                return all_scores, agents
            agents = self.generate_new_population(population_size, agents_sorted)
                
        return all_scores, agents
    
GA = GeneticEvolution(maximize = True, chromosome_len = 16)
scores, agents = GA.Evolve(100000, 1000)