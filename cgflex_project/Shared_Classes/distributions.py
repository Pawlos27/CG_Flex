



from abc import ABCMeta, abstractmethod

import numpy as np
import scipy.stats as stats
from typing import Any, List, Type
import random
import matplotlib.pyplot as plt



class IDistributions(metaclass=ABCMeta): # used in 1: Nodemaker 2:IErrordistribution #i   # while plotting the truncated distributions should be shifted up by the truncated quota
    @abstractmethod
    def get_value_from_distribution(self):
     """Interface Method"""
    def get_array_from_distribution(self, size):
     """Interface Method"""
    
    def plot_distribution(self, label=""):
        pass




class Distribution_uniform(IDistributions):
    def __init__(self, min:float, max=float):
        self.min = min
        self.max = max
    def get_value_from_distribution(self):
        x = np.random.uniform(low = self.min, high = self.max)
        return x
    def get_array_from_distribution(self, size):
        x = np.random.uniform(low = self.min, high = self.max, size=size)
        return x
    def plot_distribution(self, label=""):
        x = [self.min,self.max]
        y = [1,1]
                    
        plt.plot(x, y)
        plt.title(label + f": {self.__class__.__name__}")
        plt.show()


class Distribution_normal(IDistributions):
    def __init__(self, mu:float, sigma:float):
        self.mu = mu
        self.sigma = sigma
    def get_value_from_distribution(self):
        x = np.random.normal(self.mu,self.sigma)
        return x
    def get_array_from_distribution(self, size):
        x = np.random.normal(self.mu,self.sigma,size=size)
    def plot_distribution(self, label=""):
        sigma = self.sigma
        mu=self.mu
        x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
        plt.plot(x, stats.norm.pdf(x, mu, sigma))
        plt.title(label + f": {self.__class__.__name__}")
        plt.show()

class Distribution_normal_truncated_at_3sigma(IDistributions):
    def __init__(self, mu:float, sigma:float):
        self.mu = mu
        self.sigma = sigma
    def get_value_from_distribution(self):
        x = np.random.normal(self.mu,self.sigma)
        lower_border = self.mu-(3*self.sigma)
        upper_border = self.mu+(3*self.sigma)
        while x< lower_border or x> upper_border:
            x = np.random.normal(self.mu,self.sigma)
        return x
    def get_array_from_distribution(self, size):
        x = np.random.normal(self.mu,self.sigma,size=1)
        lower_border = self.mu-(3*self.sigma)
        upper_border = self.mu+(3*self.sigma)
        while x< lower_border or x> upper_border:
            x = np.random.normal(self.mu,self.sigma,size=1)
        for j in range(size-1):
            y = np.random.normal(self.mu,self.sigma,size=1)
            while y< lower_border or y> upper_border:
                y = np.random.normal(self.mu,self.sigma,size=1)
            x= np.append(x, y)
        return x
    def plot_distribution(self, label=""):
        sigma = self.sigma
        mu=self.mu
        x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
        plt.plot(x, stats.norm.pdf(x, mu, sigma))
        plt.title(label + f": {self.__class__.__name__}")
        plt.show()

class Distribution_mixture_of_normals(IDistributions):
    def __init__(self, list_of_mus:List[float], sigma:float):
        self.list_of_mus = list_of_mus
        self.sigma = sigma
    def get_value_from_distribution(self):
        mu= random.choice(self.list_of_mus)
        x = np.random.normal(mu,self.sigma)
        return x
    
    def get_array_from_distribution(self, size):
        mu= random.choice(self.list_of_mus)
        x = np.random.normal(mu,self.sigma, size=1)
        for j in range(size-1):
            mu= random.choice(self.list_of_mus)
            y = np.random.normal(mu,self.sigma,size=1)
            x= np.append(x, y)
        return x

    def plot_distribution(self, label=""):
        x = np.linspace(self.list_of_mus[0] - 3*self.sigma, self.list_of_mus[-1]+ 3*self.sigma, 100)
        number_distributions = len(self.list_of_mus)
        sum_distribuitons = 0
        for mu in self.list_of_mus:
            y = stats.norm.pdf(x, mu, self.sigma)/number_distributions
            sum_distribuitons = sum_distribuitons  + y
            plt.plot(x,y)
        plt.plot(x, sum_distribuitons)
        plt.title(label + f": {self.__class__.__name__}")
        plt.show()

class Distribution_mixture_of_normals_truncated_at_3sigma(IDistributions):
    def __init__(self, list_of_mus:List[float], sigma:float):
        self.list_of_mus = list_of_mus
        self.sigma = sigma
    def get_value_from_distribution(self):
        mu= random.choice(self.list_of_mus)
        x = np.random.normal(mu,self.sigma)
        lower_border = self.list_of_mus[0]-(3*self.sigma)
        upper_border = self.list_of_mus[-1]+(3*self.sigma)
        while x< lower_border or x> upper_border:
            x = np.random.normal(mu,self.sigma)
        return x
    
    def get_array_from_distribution(self, size):
        mu= random.choice(self.list_of_mus)
        x = np.random.normal(mu,self.sigma, size=1)
        lower_border = self.list_of_mus[0]-(3*self.sigma)
        upper_border = self.list_of_mus[-1]+(3*self.sigma)
        while x< lower_border or x> upper_border:
            x = np.random.normal(mu,self.sigma,size=1)
        for j in range(size-1):
            mu= random.choice(self.list_of_mus)
            y = np.random.normal(mu,self.sigma,size=1)
            while y< lower_border or y> upper_border:
                y = np.random.normal(mu,self.sigma,size=1)
            x= np.append(x, y)

        return x

    def plot_distribution(self, label=""):
        x = np.linspace(self.list_of_mus[0] - 3*self.sigma, self.list_of_mus[-1]+ 3*self.sigma, 100)
        number_distributions = len(self.list_of_mus)
        sum_distribuitons = 0
        for mu in self.list_of_mus:
            y = stats.norm.pdf(x, mu, self.sigma)/number_distributions
            sum_distribuitons = sum_distribuitons  + y
            plt.plot(x,y)
        plt.plot(x, sum_distribuitons)
        plt.title(label + f": {self.__class__.__name__}")
        plt.show()

class Distribution_mixture_of_normals_truncated_custom(IDistributions): # mus of mixture need to be within the borders
    def __init__(self, list_of_mus:List[float], sigma:float, lower_border=0.0, upper_border=1.0):
        self.sigma = sigma
        self.lower_border = lower_border
        self.upper_border = upper_border
        for mu in list_of_mus:
            if not (lower_border-6*sigma <= mu <= upper_border+6*sigma):
                raise ValueError("All values in list_of_mus must be between lower_border and upper_border")
        self.list_of_mus = list_of_mus
    def get_value_from_distribution(self):
        mu= random.choice(self.list_of_mus)
        x = np.random.normal(mu,self.sigma)
        lower_border = self.lower_border
        upper_border = self.upper_border
        while x< lower_border or x> upper_border:
            x = np.random.normal(mu,self.sigma)
        return x
    
    def get_array_from_distribution(self, size):
        mu= random.choice(self.list_of_mus)
        x = np.random.normal(mu,self.sigma,size=1)
        lower_border = self.lower_border
        upper_border = self.upper_border
        while x< lower_border or x> upper_border:
            x = np.random.normal(mu,self.sigma, size=1)
        for j in range(size-1):
            mu= random.choice(self.list_of_mus)
            y = np.random.normal(mu,self.sigma,size=1)
            while y< lower_border or y> upper_border:
                y = np.random.normal(mu,self.sigma, size=1)
            x= np.append(x, y)
        return x
    
    def plot_distribution(self, label=""):
        x = np.linspace(self.lower_border, self.upper_border, 100)
        number_distributions = len(self.list_of_mus)
        sum_distribuitons = 0
        for mu in self.list_of_mus:
            y = stats.norm.pdf(x, mu, self.sigma)/number_distributions
            sum_distribuitons = sum_distribuitons  + y
            plt.plot(x,y)
        plt.plot(x, sum_distribuitons)
        plt.title(label + f": {self.__class__.__name__}")
        plt.show()



class Distrib_custom_with_valuelist(IDistributions):
    def __init__(self, valuelist= List[Type[float]] ):
        self.valuelist = valuelist
    def get_value_from_distribution(self):
        x = random.choice(self.valuelist)
        return x
    def get_array_from_distribution(self, size):
        x = random.choice(self.valuelist)
        for j in range(size-1):
            y = random.choice(self.valuelist)
            x= np.append(x, y)
        return x
    def plot_distribution(self, label="distribution"):
        pass







if __name__ == "__main__":

    #x= Distribution_uniform(max=1,min=0)
    #x= Distribution_normal(mu=0.18, sigma=0.06)
    #x= Distribution_normal_truncated_at_3sigma(mu=0.5, sigma=0.1)
    #x= Distribution_mixture_of_normals(list_of_mus=[-0.1 , 0.1,],sigma=0.03)
    #x = Distribution_mixture_of_normals(list_of_mus=[0.1,0.15], sigma= 0.03)
    x= Distribution_mixture_of_normals(list_of_mus=[0,0.15,0.3,0.45, 0.6,0.75,0.9, 1.05,1.2,1.35,1.5,1.65,1.8,1.95,2.1,2.25,2.4,2.55,2.7,2.85],sigma=0.1)
    #x= Distribution_mixture_of_normals(list_of_mus=[0.1,0.2,0.3,0.6,0.7,0.8],sigma=0.1)
    #x= Distribution_mixture_of_normals_truncated_custom(sigma=0.2, list_of_mus=[0.9,0.7,1.2,1,0.9,1.1,1.2,0.8])
    #x =  Distribution_mixture_of_normals_truncated_custom(list_of_mus=[-0.075,0.0, 0.075,  0.15,  0.225,   0.3,   0.375,   0.7,   0.775,  0.85, 0.925, 1, 1.075], sigma=0.05)
    x.plot_distribution(label="uniform abstraction")
   
