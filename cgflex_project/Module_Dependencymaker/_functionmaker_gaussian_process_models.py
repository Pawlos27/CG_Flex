
from abc import ABCMeta, abstractmethod
import GPy


class IGaussianProcessModel(metaclass=ABCMeta):
    """
    An abstract base class that defines the interface for Gaussian Process Models.
    This interface provides methods for training the model, making predictions, and
    accessing model-specific properties.
    """

    @abstractmethod
    def train(self, X, Y, kernel):
        """
        Trains the Gaussian Process Model with the given data and kernel.

        Args:
            X (np.ndarray): Input features for training.
            Y (np.ndarray): Output/target values for training.
            kernel: The kernel to be used in the Gaussian Process.
        """
        pass

    @abstractmethod
    def predict(self, X_new):
        """
        Makes predictions using the Gaussian Process Model for new input data.

        Args:
            X_new (np.ndarray): New input data for which predictions are required.

        Returns:
            np.ndarray: Predicted values for the input data.
        """
        pass

    @abstractmethod
    def return_kernel_dimensions(self)-> int:
        """
        Returns the number of input dimensions of the kernel used in the model.

        Returns:
            int: The number of input dimensions.
        """
        pass

    @abstractmethod
    def posterior_samples_f(self,X, size):
        """
        Generates posterior samples from the Gaussian Process Model.

        Args:
            X (np.ndarray): Input data for generating samples.
            size (int): Number of samples to generate.

        Returns:
            np.ndarray: Posterior samples.
        """
        pass


class GPyModel(IGaussianProcessModel):
    """
    An implementation of the IGaussianProcessModel interface using the GPy library.
    This class encapsulates a GPy Gaussian Process model and provides functionalities
    for training, prediction, and accessing model properties.

    Attributes:
        model: The underlying GPy model instance.
    """
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
