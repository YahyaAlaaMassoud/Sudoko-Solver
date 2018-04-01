from random import randint
import numpy as np

class Individual():
    def __init__(self, length, sqrt_length):
        self.__data = [randint(1, sqrt_length) for x in range(length)]
        self.__sqrt_length = sqrt_length
        self.__age = 0
        self.__calculate_fitness()
        
    def __hash__(self):
        return hash(str(self.__data))
        
    def __lt__(self, other):
        return self.__fitness < other.__fitness
    
    def __eq__(self, other):
        return self.__fitness == other.__fitness
    
    def __gt__(self, other):
        return self.__fitness > other.__fitness
    
    def get_fitness(self):
        return self.__fitness
    
    def set_data(self, data):
        self.__data = data
        self.__calculate_fitness()
    
    def get_data(self):
        return self.__data
    
    def get_age(self):
        return self.__age
    
    def increment_age(self):
        self.__age += 1
        
    def __calculate_fitness(self):
        data = np.array(self.__data).reshape(self.__sqrt_length, self.__sqrt_length)
        fitness_row = 0.
        fitness_col = 0.
        for i in range(int(np.sqrt(len(self.__data)))):
            row = data[i, :]
            fitness_row += ((1. / (len(row) + 1 - len(np.unique(row)))) / self.__sqrt_length)
            col = data[:, i]
            fitness_col += ((1. / (len(col) + 1 - len(np.unique(col)))) / self.__sqrt_length)
        fitness_unique_box = 0.
        inc = np.int(np.sqrt(self.__sqrt_length))
        for i in range(0, self.__sqrt_length, inc):
            for j in range(0, self.__sqrt_length, inc):
                box = data[i: i + inc, j: j + inc]
                fitness_unique_box += ((1. / (inc * inc + 1 - len(np.unique(box)))) / self.__sqrt_length)
        self.__fitness = fitness_row * fitness_col * fitness_unique_box
