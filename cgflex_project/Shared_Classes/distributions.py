



from abc import ABCMeta, abstractmethod

import numpy as np
import scipy.stats as stats
from typing import Any, List, Type, Union
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
    def __init__(self, min:float, max:float):
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

class Distribution_uniform_random_at_construction(IDistributions):
    def __init__(self, min_absolute:float, max_absolute:float):
        a = np.random.uniform(low = min_absolute, high = max_absolute)
        b = np.random.uniform(low = min_absolute, high = max_absolute)
        while a == b :
            a = np.random.uniform(low = min_absolute, high = max_absolute)
            b = np.random.uniform(low = min_absolute, high = max_absolute)

        if a < b : 
            self.min = a
            self.max = b
        else :
            self.min = b
            self.max = a
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

class Distribution_uniform_random_at_every_acess(IDistributions):
    def __init__(self, min_absolute:float, max_absolute:float):
        self.min_absolute = min_absolute
        self.max_absolute = max_absolute

        self.min = self.min_absolute
        self.max = self.max_absolute
    def get_value_from_distribution(self):
        a = np.random.uniform(low = self.min_absolute, high = self.max_absolute)
        b = np.random.uniform(low = self.min_absolute, high = self.max_absolute)
        while a == b :
            a = np.random.uniform(low = self.min_absolute, high = self.max_absolute)
            b = np.random.uniform(low = self.min_absolute, high = self.max_absolute)
        if a < b : 
            self.min = a
            self.max = b
        else :
            self.min = b
            self.max = a
        x = np.random.uniform(low = self.min, high = self.max)
        return x
    def get_array_from_distribution(self, size):
        a = np.random.uniform(low = self.min_absolute, high = self.max_absolute)
        b = np.random.uniform(low = self.min_absolute, high = self.max_absolute)
        while a == b :
            a = np.random.uniform(low = self.min_absolute, high = self.max_absolute)
            b = np.random.uniform(low = self.min_absolute, high = self.max_absolute)
        if a < b : 
            self.min = a
            self.max = b
        else :
            self.min = b
            self.max = a
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

class Distribution_normal_random_sigma_at_construction(IDistributions):
    def __init__(self, mu:float, sigma_max:float):
        self.mu = mu
        self.sigma = np.random.uniform(low = 0, high = sigma_max)
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

class Distribution_normal_random_all_inside_borders_random_at_construction(IDistributions):
    def __init__(self,lower_border=0 , upper_border=1 ):
        sigma_max = (abs(upper_border - lower_border))/6
        self.sigma = np.random.uniform(low = 0, high = sigma_max)
        self.mu =  np.random.uniform(low = lower_border+3*self.sigma, high = upper_border-3*self.sigma)

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

class Distribution_normal_random_sigma_at_acess(IDistributions):
    def __init__(self, mu:float, sigma_max:float):
        self.sigma_max = sigma_max
        self.mu = mu
        self.sigma = np.random.uniform(low = 0, high = sigma_max)
    def get_value_from_distribution(self):
        self.sigma = np.random.uniform(low = 0, high = self.sigma_max)
        x = np.random.normal(self.mu,self.sigma)
        return x
    def get_array_from_distribution(self, size):
        self.sigma = np.random.uniform(low = 0, high = self.sigma_max)
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


class Distribution_normal_truncated_at_3sigma_random_all_inside_borders_random_at_construction(IDistributions):
    def __init__(self,lower_border=0 , upper_border=1 ):
        sigma_max = (abs(upper_border - lower_border))/6
        self.sigma = np.random.uniform(low = 0, high = sigma_max)
        self.mu =  np.random.uniform(low = lower_border+3*self.sigma, high = upper_border-3*self.sigma)

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

class Distribution_normal_truncated_at_3sigma_random_sigma_at_construction(IDistributions):
    def __init__(self, mu:float, sigma_max:float):
        self.mu = mu
        self.sigma = np.random.uniform(low = 0, high = sigma_max)
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

class Distribution_normal_truncated_at_3sigma_random_sigma_at_acess(IDistributions):
    def __init__(self, mu:float, sigma_max:float):
        self.mu = mu
        self.sigma_max = sigma_max
        self.sigma = sigma_max
    def get_value_from_distribution(self):
        self.sigma = np.random.uniform(low = 0, high = self.sigma_max)
        x = np.random.normal(self.mu,self.sigma)
        lower_border = self.mu-(3*self.sigma)
        upper_border = self.mu+(3*self.sigma)
        while x< lower_border or x> upper_border:
            x = np.random.normal(self.mu,self.sigma)
        return x
    def get_array_from_distribution(self, size):
        self.sigma = np.random.uniform(low = 0, high = self.sigma_max)
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

class Distribution_normal_truncated_at_3sigma_bound_to_zero(IDistributions):
    def __init__(self, sigma:float):
        
        self.sigma = sigma
        self.mu = 3*self.sigma
    def get_value_from_distribution(self):
        x = np.random.normal(self.mu,self.sigma)
        lower_border = 0
        upper_border = self.mu+(3*self.sigma)
        while x< lower_border or x> upper_border:
            x = np.random.normal(self.mu,self.sigma)
        return x
    def get_array_from_distribution(self, size):
        x = np.random.normal(self.mu,self.sigma,size=1)
        lower_border = 0
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

class Distribution_normal_truncated_at_3sigma_bound_to_zero_random_at_construction(IDistributions):
    def __init__(self, sigma_max:float):
        
        self.sigma = np.random.uniform(low = 0, high = sigma_max)
        self.mu = 3*self.sigma
    def get_value_from_distribution(self):
        x = np.random.normal(self.mu,self.sigma)
        lower_border = 0
        upper_border = self.mu+(3*self.sigma)
        while x< lower_border or x> upper_border:
            x = np.random.normal(self.mu,self.sigma)
        return x
    def get_array_from_distribution(self, size):
        x = np.random.normal(self.mu,self.sigma,size=1)
        lower_border = 0
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
        self.list_of_mus.sort()
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

class Distribution_mixture_of_normals_random_uniform_at_construction(IDistributions):
    def __init__(self,  sigma:float, lower_border=0.0, upper_border=1.0, components=20):
        self.sigma = sigma
        self.list_of_mus = np.random.uniform(lower_border , upper_border , components).tolist()
        self.list_of_mus.sort()
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
        self.list_of_mus.sort()
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

class Distribution_mixture_of_normals_truncated_at_3sigma_and_outlier_correction_for_interpolation(IDistributions):
    def __init__(self, list_of_mus:List[float], sigma:float, upper_limit:float):
        self.list_of_mus = list_of_mus
        self.list_of_mus.sort()
        self.sigma = sigma

        
        self.upper_border = upper_limit
        self.lower_border = self.list_of_mus[0]-(3*self.sigma)
        if self.lower_border < 0 :
            self.lower_border = 0

    def get_value_from_distribution(self):
        mu= random.choice(self.list_of_mus)
        x = np.random.normal(mu,self.sigma)
        while x< self.lower_border or x> self.upper_border:
            x = np.random.normal(mu,self.sigma)
        return x
    
    def get_array_from_distribution(self, size):
        mu= random.choice(self.list_of_mus)
        x = np.random.normal(mu,self.sigma, size=1)
        while x< self.lower_border or x> self.upper_border:
            x = np.random.normal(mu,self.sigma,size=1)
        for j in range(size-1):
            mu= random.choice(self.list_of_mus)
            y = np.random.normal(mu,self.sigma,size=1)
            while y< self.lower_border or y> self.upper_border:
                y = np.random.normal(mu,self.sigma,size=1)
            x= np.append(x, y)
        return x

    def plot_distribution(self, label=""):
        x = np.linspace( self.lower_border, self.upper_border, 100)
        number_distributions = len(self.list_of_mus)
        sum_distribuitons = 0
        for mu in self.list_of_mus:
            y = stats.norm.pdf(x, mu, self.sigma)/number_distributions
            sum_distribuitons = sum_distribuitons  + y
            plt.plot(x,y)
        plt.plot(x, sum_distribuitons)
        plt.title(label + f": {self.__class__.__name__}")
        plt.show()


class Distribution_mixture_of_normals_truncated_at_3sigma_random_sigma_outward_random_inside_borders_and_uniform_at_construction(IDistributions):
    def __init__(self,  lower_border=0.0, upper_border=1.0, components=20):
        self.sigma = np.random.uniform( 0 , upper_border/10 )
        self.list_of_mus = np.random.uniform(lower_border + 3*self.sigma , upper_border - 3*self.sigma, components).tolist()
        self.list_of_mus.sort()
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


class Distribution_mixture_of_normals_truncated_at_3sigma_outward_random_inside_borders_and_uniform_at_construction(IDistributions):
    def __init__(self,  sigma:float, lower_border=0.0, upper_border=1.0, components=20):
        self.sigma = sigma
        self.list_of_mus = np.random.uniform(lower_border + 3*sigma , upper_border - 3*sigma, components).tolist()
        self.list_of_mus.sort()
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

class Distribution_mixture_of_normals_truncated_at_3sigma_inwards_random_uniform_at_construction(IDistributions):
    def __init__(self,  sigma:float, lower_border=0.0, upper_border=1.0, components=20):
        self.sigma = sigma
        self.upper_border = upper_border
        self.lower_border = lower_border
        self.list_of_mus = np.random.uniform(lower_border - 2.5*sigma , upper_border + 2.5*sigma , components).tolist()
        self.list_of_mus.sort()
    def get_value_from_distribution(self):
        mu= random.choice(self.list_of_mus)
        x = np.random.normal(mu,self.sigma)

        while x< self.lower_border or x> self.upper_border:
            x = np.random.normal(mu,self.sigma)
        return x
    
    def get_array_from_distribution(self, size):
        mu= random.choice(self.list_of_mus)
        x = np.random.normal(mu,self.sigma, size=1)
        while x< self.lower_border or x> self.upper_border:
            x = np.random.normal(mu,self.sigma,size=1)
        for j in range(size-1):
            mu= random.choice(self.list_of_mus)
            y = np.random.normal(mu,self.sigma,size=1)
            while y< self.lower_border or y> self.upper_border:
                y = np.random.normal(mu,self.sigma,size=1)
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

class Distribution_mixture_of_normals_truncated_at_3sigma_inwards_random_mu_and_random_spread_uniform_at_construction(IDistributions):
    def __init__(self,  sigma:float, lower_border_max=0.0, upper_border_max=1.0, components=2):
        self.sigma = sigma
        limits= np.random.uniform(lower_border_max , upper_border_max , 2).tolist()
        limits.sort()
        self.upper_border = limits[1]
        self.lower_border = limits[0]
        self.list_of_mus = np.random.uniform(self.lower_border - 2.5*sigma , self.upper_border + 2.5*sigma , components).tolist()
        self.list_of_mus.sort()
    def get_value_from_distribution(self):
        mu= random.choice(self.list_of_mus)
        x = np.random.normal(mu,self.sigma)

        while x< self.lower_border or x> self.upper_border:
            x = np.random.normal(mu,self.sigma)
        return x
    
    def get_array_from_distribution(self, size):
        mu= random.choice(self.list_of_mus)
        x = np.random.normal(mu,self.sigma, size=1)
        while x< self.lower_border or x> self.upper_border:
            x = np.random.normal(mu,self.sigma,size=1)
        for j in range(size-1):
            mu= random.choice(self.list_of_mus)
            y = np.random.normal(mu,self.sigma,size=1)
            while y< self.lower_border or y> self.upper_border:
                y = np.random.normal(mu,self.sigma,size=1)
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






class IErrorterm_mixture_distance_distribution_collection(metaclass=ABCMeta):
    distribution_list: List[Union[Distribution_uniform, 
                                  Distribution_uniform_random_at_construction,
                                  Distribution_mixture_of_normals_truncated_at_3sigma,
                                  Distribution_mixture_of_normals_truncated_custom,
                                  Distribution_mixture_of_normals_truncated_at_3sigma_inwards_random_mu_and_random_spread_uniform_at_construction,
                                  Distribution_mixture_of_normals_truncated_at_3sigma_random_sigma_outward_random_inside_borders_and_uniform_at_construction]]
    """we cannot allow all IDistribution Classes, to prevent cirucular dependencies"""

    @abstractmethod
    def get_distance_array(self, size:int,maximum_distance:float ):
        """Interface Method"""

class Collection_of_mixture_distance_distribution_random(IErrorterm_mixture_distance_distribution_collection):
    def __init__(self):
        self.distribution_list=[Distribution_uniform_random_at_construction(max_absolute=1,min_absolute=0),
                                Distribution_uniform(min=0.9, max=1),
                                Distribution_mixture_of_normals_truncated_custom(list_of_mus=[1], sigma=0.00),
                                Distribution_mixture_of_normals_truncated_custom(list_of_mus=[0.1], sigma=0.1,lower_border=0.25, upper_border=0.55),
                                Distribution_uniform(min=0.1, max=0.1),
                                Distribution_normal_truncated_at_3sigma_random_all_inside_borders_random_at_construction(),
                                Distribution_mixture_of_normals_truncated_custom(list_of_mus=[0.9,1,1.1,], sigma=0.7),
                                Distribution_mixture_of_normals_truncated_custom(list_of_mus=[0.9,1,1.1,], sigma=0.2),
                                Distribution_mixture_of_normals_truncated_custom(list_of_mus=[0.4], sigma=1),
                                Distribution_mixture_of_normals_truncated_at_3sigma_inwards_random_mu_and_random_spread_uniform_at_construction(lower_border_max=0, upper_border_max=0.3, sigma= 0.1), 
                                Distribution_mixture_of_normals_truncated_at_3sigma_inwards_random_mu_and_random_spread_uniform_at_construction(lower_border_max=0.7, upper_border_max=1, sigma= 0.1),
                                Distribution_mixture_of_normals_truncated_at_3sigma_inwards_random_mu_and_random_spread_uniform_at_construction(sigma= 0.1), 
                                Distribution_mixture_of_normals_truncated_at_3sigma_random_sigma_outward_random_inside_borders_and_uniform_at_construction()]
  
    def get_distance_array(self, size,maximum_distance):
       random_distribution = random.choice(self.distribution_list)
       distance_array = random_distribution.get_array_from_distribution(size=size)
       distance_array = distance_array*maximum_distance
       return distance_array
    
class Collection_of_mixture_distance_distribution_uniform(IErrorterm_mixture_distance_distribution_collection):
    def __init__(self):
        self.distribution_list=[Distribution_uniform(min=0.5, max=0.6), Distribution_uniform(min=0.9, max=1), Distribution_uniform_random_at_construction(max_absolute=1,min_absolute=0)] 
    def get_distance_array(self, size,maximum_distance):
       random_distribution = random.choice(self.distribution_list)
       distance_array = random_distribution.get_array_from_distribution(size=size)
       distance_array = distance_array*maximum_distance
       return distance_array

class Collection_of_mixture_distance_distribution_trucated_inwards_steep(IErrorterm_mixture_distance_distribution_collection):
    def __init__(self):
        self.distribution_list=[Distribution_mixture_of_normals_truncated_at_3sigma_inwards_random_mu_and_random_spread_uniform_at_construction(lower_border_max=0, upper_border_max=0.3, sigma= 0.1), Distribution_mixture_of_normals_truncated_at_3sigma_inwards_random_mu_and_random_spread_uniform_at_construction(lower_border_max=0.7, upper_border_max=1, sigma= 0.1)]
    def get_distance_array(self, size,maximum_distance):
       distribution = random.choice(self.distribution_list)
       distance_array = distribution.get_array_from_distribution(size=size)
       distance_array = distance_array*maximum_distance
       return distance_array




class Complex_distribution_list_of_Mus_maker:
    def __init__(self,distribution_for_distances_between_components: IErrorterm_mixture_distance_distribution_collection = Collection_of_mixture_distance_distribution_random(), number_of_maximum_modes=5 ,  total_number_of_elements= 20,   maximum_distance_between_modes_factor=5, multimodal_normal = False, fixed_number_of_modes= False) -> None:
        if number_of_maximum_modes <=0:
            raise ValueError("number of modes must be >0")
        else:
            pass
        if number_of_maximum_modes > total_number_of_elements:
            raise ValueError("number of modes must be <= total_number_of_elements")
        self.maximum_distance_between_modes_factor = maximum_distance_between_modes_factor
        self.total_number_of_elements = total_number_of_elements
        self.number_of_maximum_modes = number_of_maximum_modes
        self.distribution_for_distances_between_components = distribution_for_distances_between_components
        self.multimodal_normal = multimodal_normal
        self.fixed_number_of_modes = fixed_number_of_modes

    def return_sigma(self)-> float:
        sigma = self.sigma
        return sigma
    def sample_lists_of_mus(self, size_of_samples:int ,  maximum_total_deviation:float)-> List[List[float]]:
        lists_of_mus = []
        self.maximum_total_deviation = maximum_total_deviation
        for i in range(size_of_samples):
            if self.fixed_number_of_modes == True:
                modes = self.number_of_maximum_modes
            else: 
                modes = random.randint(2, self.number_of_maximum_modes)
            self._calculate_maximum_distance_for_mixture_model(modes=modes)
            self._make_list_for_components_per_mode(modes=modes)
            single_list_of_mus = self._make_list_of_mus_for_mixture_model(modes=modes)
            lists_of_mus.append(single_list_of_mus)
        return lists_of_mus

    def _calculate_maximum_distance_for_mixture_model(self,modes:int): # max distances between elements, + borders(4)+ distances between modes
        denominator = (self.total_number_of_elements-1)+4+(modes*(self.maximum_distance_between_modes_factor-1))
        maximum_distance = self.maximum_total_deviation/denominator
        self.maximum_distance = maximum_distance
        self.sigma = maximum_distance/1.5

    def _make_list_for_components_per_mode(self, modes:int):
        if self.multimodal_normal == True :
            self.components_per_mode = [1 for _ in range(modes)]
        else:
            components_per_mode = [1 for _ in range(modes)]
            # Distribute the remaining elements (if any) after each mode has at least 1
            remaining_elements = self.total_number_of_elements - modes
            for i in range(remaining_elements):
                components_per_mode[i % modes] += 1
            self.components_per_mode = components_per_mode
            

    def _make_list_of_mus_for_mixture_model(self, modes:int ) ->List[float]:

        components_per_mode = self.components_per_mode
        maximum_distance = self.maximum_distance
        maximum_distance_between_modes_factor = self.maximum_distance_between_modes_factor
        distribution = self.distribution_for_distances_between_components
        list_of_mus = [maximum_distance * 2]
        for mode_index in range(modes):
            elements_in_mode = components_per_mode[mode_index]
            if elements_in_mode > 1:
                size_descending_array = random.randint(1, elements_in_mode - 1)
                size_ascending_array = elements_in_mode - 1 - size_descending_array
                distances_array_descending = distribution.get_distance_array(size=size_descending_array, maximum_distance=maximum_distance)
                distances_array_descending = np.sort(distances_array_descending)[::-1]
                distances_array_ascending = distribution.get_distance_array(size=size_ascending_array, maximum_distance=maximum_distance)
                distances_array_ascending = np.sort(distances_array_ascending)
                for i in distances_array_descending:
                    new_mu = list_of_mus[-1] + i
                    list_of_mus.append(new_mu)
                for i in distances_array_ascending:
                    new_mu = list_of_mus[-1] + i
                    list_of_mus.append(new_mu)
            if mode_index < modes - 1:
                distance_between_modes_factor = np.random.uniform(low=1.5, high=maximum_distance_between_modes_factor)
                x = list_of_mus[-1] + (maximum_distance * distance_between_modes_factor)
                list_of_mus.append(x)
        list_of_mus = list_of_mus[:self.total_number_of_elements]
        # stretch and correct, 2 times because sigma has iterative effect, at first round slightly to big at second time, sligtly to small, we aim for to small
        for i in range(2):
            stretch_factor = ((self.maximum_total_deviation - self.sigma*3 ) - 0) / (list_of_mus[-1] - 0)
            list_of_mus = [mu * stretch_factor for mu in list_of_mus]
            self.sigma = self.sigma*stretch_factor
        return list_of_mus
        



class Distribution_mixture_of_normals_controlled_modes_complex_spread_of_mus_and_random(IDistributions):
    def __init__(self, mixture_list_of_mus_maker : Complex_distribution_list_of_Mus_maker = Complex_distribution_list_of_Mus_maker(), lower_border = 0 , upper_border = 1):

        maximum_total_deviation = upper_border - lower_border
        self.lists_of_mus =mixture_list_of_mus_maker.sample_lists_of_mus(size_of_samples=1,maximum_total_deviation=maximum_total_deviation)
        self.sigma = mixture_list_of_mus_maker.return_sigma()
        self.distribution = Distribution_mixture_of_normals_truncated_custom(list_of_mus=self.lists_of_mus[0], sigma=self.sigma, lower_border= lower_border, upper_border=upper_border)



    def get_value_from_distribution(self):
        x = self.distribution.get_value_from_distribution()
        return x
    
    def get_array_from_distribution(self, size):
        x = self.distribution.get_array_from_distribution(size=size)
        return x

    def plot_distribution(self, label=""):
        self.distribution.plot_distribution(label=label)




if __name__ == "__main__":

    #x= Distribution_uniform(max=1,min=0)
    #x= Distribution_normal(mu=0.18, sigma=0.06)
    #x= Distribution_normal_truncated_at_3sigma(mu=0.5, sigma=0.1)
    #x= Distribution_mixture_of_normals(list_of_mus=[-0.1 , 0.1,],sigma=0.03)
    #x = Distribution_mixture_of_normals(list_of_mus=[0.1,0.15], sigma= 0.03)
    #x= Distribution_mixture_of_normals(list_of_mus=[0,0.15,0.3,0.45, 0.6,0.75,0.9, 1.05,1.2,1.35,1.5,1.65,1.8,1.95,2.1,2.25,2.4,2.55,2.7,2.85],sigma=0.1)
    #x= Distribution_mixture_of_normals(list_of_mus=[0.1,0.2,0.3,0.6,0.7,0.8],sigma=0.1)
    #x= Distribution_mixture_of_normals_truncated_custom(sigma=0.2, list_of_mus=[0.9,0.7,1.2,1,0.9,1.1,1.2,0.8])
    #x =  Distribution_mixture_of_normals_truncated_custom(list_of_mus=[-0.075,0.0, 0.075,  0.15,  0.225,   0.3,   0.375,   0.7,   0.775,  0.85, 0.925, 1, 1.075], sigma=0.05)
    
    
    #x= Distribution_uniform_random_at_construction(max_absolute=1,min_absolute=0)
    #x= Distribution_uniform_random_at_every_acess(max_absolute=1,min_absolute=0)
    #x= Distribution_normal_random_sigma_at_construction(sigma_max=0.1, mu= 0.5)
    #x= Distribution_normal_truncated_at_3sigma_bound_to_zero_random_at_construction(sigma_max=0.3)
    #x= Distribution_mixture_of_normals_random_uniform_at_construction(sigma= 0.1 )
    #x= Distribution_mixture_of_normals_truncated_at_3sigma_random_uniform_at_construction(sigma= 0.1 )
    #x= Distribution_mixture_of_normals_truncated_at_3sigma_inwards_random_uniform_at_construction(sigma= 0.1 )
    #x= Distribution_mixture_of_normals_truncated_at_3sigma_inwards_random_mu_and_random_spread_uniform_at_construction(sigma= 0.1)
    #x = Distribution_mixture_of_normals_truncated_at_3sigma_outward_random_inside_borders_and_uniform_at_construction( sigma=0.1)
   
    #x = Distribution_normal_truncated_at_3sigma_random_all_inside_borders_random_at_construction()
    ##x = Distribution_mixture_of_normals_controlled_modes_complex_spread_of_mus_and_random(mixture_list_of_mus_maker= Complex_distribution_list_of_Mus_maker(total_number_of_elements=14,distribution_for_distances_between_components=Collection_of_mixture_distance_distribution_trucated_inwards_steep()))
    x= Distribution_mixture_of_normals_truncated_at_3sigma_outward_random_inside_borders_and_uniform_at_construction(sigma= 0.07, components= 9)
    #x = Distribution_mixture_of_normals_controlled_modes_complex_spread_of_mus_and_random(mixture_list_of_mus_maker= Complex_distribution_list_of_Mus_maker())
    #x = Distribution_mixture_of_normals_truncated_at_3sigma_inwards_random_uniform_at_construction(sigma= 0.08, components= 11)] 


    #x.get_array_from_distribution(size=2)

    x.plot_distribution(label="uniform abstraction")
   
