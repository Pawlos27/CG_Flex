
from abc import ABCMeta, abstractmethod
import GPy


class IGaussianProcessModel(metaclass=ABCMeta):

    @abstractmethod
    def train(self, X, Y, kernel):
        pass

    @abstractmethod
    def predict(self, X_new):
        pass

    @abstractmethod
    def return_kernel_dimensions(self)-> int:
        pass

    @abstractmethod
    def posterior_samples_f(self,X, size):
        pass



# Step 2: Create the Implementation Class
class GPyModel(IGaussianProcessModel):
    def __init__(self):
        self.model = None

    def train(self, X, Y, kernel):
        self.model = GPy.models.GPRegression(X, Y, kernel)
    
    def posterior_samples_f(self,X, size):
        if not self.model:
            raise ValueError("The model has not been trained yet.")
        else:
            samples = self.model.posterior_samples_f(X=X, size=size)
            return samples

    def predict(self, x_input):
        if not self.model:
            raise ValueError("The model has not been trained yet.")
        return self.model.predict(x_input)
    def return_kernel_dimensions(self)-> int:
        kernel = self.model.kern
        input_dim = kernel.input_dim
        return input_dim
