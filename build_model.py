from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Optimizer # Imported for validation purposes

def build_model_factory(input_dim: int, output_activation: str = 'linear', output_dim: int = 1):
    """
    Factory function to create a build_model function with specified input and output dimensions.
    
    Args:
        input_dim (int): The number of input features.
        output_activation (str): The activation function for the output layer ('softmax', 'sigmoid', or 'linear').
        output_dim (int): The number of output units.
    
    Returns:
        function: A function that builds a Sequential ANN model with the specified parameters.
    """
    def build_model(neuron_layers: tuple[int] = (64, 64),
                    dropout_layers: tuple[float] = (0.2, 0.2),
                    activation: str = 'relu',
                    optimizer: str = 'adam',) -> Sequential:
        """
        Builds a Sequential ANN model with the specified parameters.
        
        Args:
            neuron_layers (tuple[int]): The shapes of the hidden layers.
            dropout_layers (tuple[float]): The dropout rates for the hidden layers.
            activation (str): The activation function to use.
            optimizer (str): The optimizer to use.
        
        Returns:
            Sequential: The built ANN model.
        
        Raises:
            ValueError: If neuron_layers is not a tuple of integers.
            ValueError: If dropout_layers is not a tuple of floats.
            ValueError: If optimizer is not a string or a Keras optimizer instance.
        """
        if not isinstance(neuron_layers, tuple) or any(not isinstance(layer, int) for layer in neuron_layers):
            raise ValueError("neuron_layers must be a tuple of integers.")
        if not isinstance(dropout_layers, tuple) or any(not isinstance(layer, float) for layer in dropout_layers):
            raise ValueError("dropout_layers must be a tuple of floats.")
        if not isinstance(optimizer, (str, Optimizer)):
            raise ValueError("Optimizer must be a string or a Keras optimizer instance.")
        # activation, and metrics are checked by Keras, they can also be non-string types

        model = Sequential()

        # Combine neuron and dropout layers
        combined_layers = []
        for neurons, dropout in zip(neuron_layers, dropout_layers):
            combined_layers.append(neurons)
            combined_layers.append(dropout)

        # Add input layer
        model.add(Dense(combined_layers[0], input_dim=input_dim, activation=activation))

        # Add hidden layers
        for i in combined_layers[1:]:
            if i >= 1:
                try: # Use error management in Dense to raise error further.
                    model.add(Dense(i, activation=activation))
                except Exception as e:
                    raise ValueError(f"An error occurred (layer): {e}") from e
            elif 0 <= i < 1: # dropout layer, cannot be i<=1.
                model.add(Dropout(i))
            else:
                raise ValueError(f"error in layer construction, invalid layer value: {i}\n" +\
                    f"value {i} should be a positive integer or float between 0 and 1.")

        # Add output layer
        model.add(Dense(output_dim, activation=output_activation))

        # Determine loss function based on output activation
        if output_activation == 'softmax':
            loss = 'categorical_crossentropy'
        elif output_activation == 'sigmoid':
            loss = 'binary_crossentropy'
        elif output_activation == 'linear':
            loss = 'mean_squared_error'
        else:
            raise ValueError(f"Invalid output activation: {output_activation}. Must be 'softmax', 'sigmoid', or 'linear'.")

        # Compile model
        try:
            model.compile(loss=loss, optimizer=optimizer)
        except ValueError as e:
            raise ValueError(f"Error compiling model: {e}") from e
        return model

    return build_model