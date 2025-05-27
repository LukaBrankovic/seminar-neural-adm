import torch
from . import initializations as init

class Switchers(object):
    """Switcher object for several utilities
    """
    _activations = {
        'relu': lambda x: torch.nn.ReLU(inplace=True),
        'tanh': lambda x: torch.nn.Tanh(),
        'gelu': lambda x: torch.nn.GELU()
    }

    # Modified lambdas to accept *args and **kwargs
    # The order of arguments in the lambda definition still matters for *args,
    # but **kwargs will capture any additional keyword arguments.
    # The get_decoder_init methods in initializations.py must be able to handle these args and kwargs.
    # Specifically, they now have precomputed_X_pca_tensor and precomputed_pca_obj as named keyword arguments.

    _initializations = {
        'kmeans': lambda epochs, batch_size, learning_rate, K, seed, init_path, name, n_components, data, device, 
                         num_gpus, hidden_size, activation, master, num_cpus, 
                         y, supervised_loss_weight, # These were the original positional/keyword args expected by the lambda
                         # Now we add **kwargs to capture any new ones like precomputed_X_pca_tensor
                         **kwargs:  # Capture additional keyword arguments
            
            init.KMeansInitialization.get_decoder_init(
                epochs, batch_size, learning_rate, K, seed, init_path, name, n_components, data, device, 
                num_gpus, hidden_size, activation, master, num_cpus,
                y, supervised_loss_weight, # Pass y and supervised_loss_weight explicitly
                **kwargs # Pass through any other keyword arguments
            ),

        'gmm': lambda epochs, batch_size, learning_rate, K, seed, init_path, name, n_components, data, device, 
                      num_gpus, hidden_size, activation, master, num_cpus,
                      y, supervised_loss_weight,
                      **kwargs: 
            
            init.GMMInitialization.get_decoder_init(
                epochs, batch_size, learning_rate, K, seed, init_path, name, n_components, data, device, 
                num_gpus, hidden_size, activation, master, num_cpus,
                y, supervised_loss_weight,
                **kwargs
            ),
        
        'random': lambda epochs, batch_size, learning_rate, K, seed, init_path, name, n_components, data, device, 
                       num_gpus, hidden_size, activation, master, num_cpus,
                       y, supervised_loss_weight,
                       **kwargs: 
            
            init.RandomInitialization.get_decoder_init(
                epochs, batch_size, learning_rate, K, seed, init_path, name, n_components, data, device, 
                num_gpus, hidden_size, activation, master, num_cpus,
                y, supervised_loss_weight,
                **kwargs
            ),
        
        'supervised': lambda epochs, batch_size, learning_rate, K, seed, init_path, name, n_components, data, device, 
                           num_gpus, hidden_size, activation, master, num_cpus, 
                           y, supervised_loss_weight, # y and supervised_loss_weight are explicitly part of supervised's signature
                           **kwargs: 
            
            init.SupervisedInitialization.get_decoder_init(
                epochs, batch_size, learning_rate, K, seed, init_path, name, n_components, data, device, 
                num_gpus, hidden_size, activation, master, num_cpus, 
                y, supervised_loss_weight, # Pass y and supervised_loss_weight explicitly
                **kwargs
            ),
    }

    @classmethod
    def get_switchers(cls) -> dict[str, object]:
        """
        Returns:
        - dict[str, object]: A dictionary where the keys are strings ('activations', 'initializations'),
          and the values are the corresponding class-level attributes.
        """
        return {
            'activations': cls._activations,
            'initializations': cls._initializations,
        }
