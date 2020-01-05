# Dependencies: Standard libraries
import numpy as np
import datetime

# Dependency: "NN_Support" (Activation, Evaluation, Plotting Functions)
from NN_Support import (sigmoid, relu, leaky_relu)

class Multilayer_NN():
    """
    class Multilayer_NN
    
    An implementation of a multilayer neural network.

    Public Methods
    * configure: Configures the NN
    * fit: Trains the NN using specified input feature data and actual output labels
    * predict: Predicts output labels based upon specified input feature data
    * get_hist: Returns the model fit history
    
    Method of Use:
    1. Instantiate the object: `model = Multilayer_NN()`
    2. Configure the model, which also initializes it: model.configure(...)
    3. Fit the model to the training data: model.fit(...)
    4. Use the model for prediction: model.predict(...)
    5. Get useful information about the model: model.get_info(...)
    """


    # ************************** STANDARD METHODS **************************
    def __init__(self, a_layers = None):
        """
        Constructor for the class
        
        Arguments:
            a_layers [OPTIONAL]: A list providing the number of nodes/units in each layer:
                        Layer 0: Number of features in input X
                        Layers 1 through L-1: Number of nodes/units in hidden layers
                        Layer L: Number of outputs in Y
   
        Returns:
            None: Values are initialized in: self._config, self._param, self._cache, self._hist
        """
        
        # Initialize parameters and other key information
        self._init_config()
        
        # If configuration info has been provided, go ahead and configure the model
        if a_layers is not None:
            self.configure(a_layers)

    def __repr__(self):
        """
        Return an "official" string representation of the object
        
        Arguments:
            None
                        
        Returns:
            String containing an "official" representation of the object
        """
        
        # Get the list of all key/value pairs in the config dictionary
        config_list = self._config

        # Return the list as a string
        retval = f"{self.__class__.__name__}({config_list})"        
        return retval
    
    def __str__(self):
        """
        Return an "friendly" string representation of the object
        
        Arguments:
            None
                        
        Returns:
            String containing a "friendly" representation of the object
        """
        
        # For now, just return whatever is generated from __repr__
        retval = self.__repr__()
        return retval

    
    # ************************** PRIVATE METHODS **************************
    def _init_config(self):
        """
        Initialize model parameters and other information
        
        Arguments:
            None
                        
        Returns:
            None: Values are initialized in: self._config, self._param, self._cache, self._hist
        """
        
        # Configuration
        # - Parameters provided associated with the configuration of the model
        #   (provided at configure or fitting)
        self._config = {
            'is_configured': False,     # Flag: The model has been configured
            'is_fitted': False,         # Flag: The model has been fitted to the training set
            
            'alpha': None,              # Learning Rate
            'lambda': None,             # Regularization Rate
            'batch_size': None,         # Size of mini-batches used for training in each iteration
            'max_epochs': None,         # Maximum number of epochs (i.e., passes through the entire training set) to allow
            'm': None,                  # Number of examples in the training set inputs
            'n_x': None,                # Number of features in the training set inputs
            'n_y': None,                # Number of outputs in the training set labels
            'L': None,                  # Total number of hidden layers + output layer
            'all_layers': [],           # Layer node/unit counts (i.e., layers 0 through L)
        }        
        
        # Parameters
        # - Coeffiecients W and b
        self._param = {
            'W': {},       # Weight coefficients for each layer 1 through L
            'b': {}       # Bias coefficients for each layer 1 through L
        }        

        # Cache
        # - values calculated during forward and backward propagation that need to be retained,
        #   including per-layer values of: Z, A, dA, dZ, dW, db
        self._cache = {
            'Z': {},       # Linear function for each layer 1 through L-1
            'A': {},       # Linear function for each layer 1 through L-1, and with A[0] = X, A[L]=Y
            'dA': {},      # Partial derivative by A; used to calculate dZ
            'dZ': {},      # Partial derivative by Z; used to calculate dW and db
            'dW': {},      # Partial derivative by W; used for update of W coefficients
            'db': {}       # Partial derivative by b; used for update of b coefficients
        }
        
        # History
        # - Fitting history and related info:
        self._hist = {
            'cost': [],              # List of cost (loss) per epoch
            'accuracy': [],          # List of training set accuracy per epoch
            'timestamp': []          # List of timestamp per epoch (as 'datetime' objects)
        }
        
    def _init_param(self):
        """
        Initialize model parameters W and b (after the model is configured)

        Arguments:
            None: Configuration info is obtained from self._config: all_layers, L
                        
        Returns:
            None: Values are initialized for each layer in self._param: W, b
        
        """
        
        # Confirm that the model is configured before attempting to initialize the parameters
        assert self._config['is_configured'], f"Cannot initialize parameters W and b until the model is configured: is_configured = {self._config['is_configured']} [{type(self._config['is_configured'])}]"

        # Make sure the parameter lists are empty
        self._param['W'] = {}    # Weight coefficients for each layer 1 through L
        self._param['b'] = {}    # Bias coefficients for each layer 1 through L

        # Loop through all hidden and the final output layer (i.e., 1 through L)
        for layer in range(1, self._config['L']+1):

            # Initialize parameters for each layer: Coeffiecients W (to random numbers) and b (to zeros)
            # NOTE: Using "He Initialization"
            W_val = np.random.randn( self._config['all_layers'][layer], self._config['all_layers'][layer-1] ) \
                            * np.sqrt( 2.0 / self._config['all_layers'][layer-1] )
            b_val = np.zeros( (self._config['all_layers'][layer], 1) )
            
            # Add the parameters to the param dictionary
            self._param['W']['W'+str(layer)] = W_val
            self._param['b']['b'+str(layer)] = b_val
            
    def _dump_info(self):
        """
        Basic dump of all internal config, parameter, cache, and history data
        
        Arguments:
            None
                        
        Returns:
            Dictionary containing internal info: self._config, self._param, self._cache, self._hist
        """
    
        # Create a dictionary containing all of the internal data dictionaries
        retval = {
            '_config': self._config,
            '_param': self._param,
            '_cache': self._cache,
            '_hist': self._hist
        }
        
        return retval

    def _set_fit_config(self, a_X, a_y, a_alpha = 0.01, a_batch_size = 32, a_max_epochs = 10000, a_lambda=0.0, a_optim='gd', a_beta1=0.9, a_beta2=0.999, a_epsilon=10e-08):
        """
        Perform fit argument checks and set attributes
        
        Arguments:
            a_X: Input features (numpy array) ~ shape( # features, # examples )
            a_y: Output labels (numpy array) ~ shape ( # outputs, # examples )
            a_alpha: Learning Rate
            a_batch_size: Number of examples to use for validation for each iteration
            a_max_epochs: Maximum number of epochs (i.e., full passes through the training set) to allow
            a_lambda: Regularization Rate (default 0.0 => No regularization)
            a_optim: Optimizer - 'none' or 'adam'
            a_beta1: Beta1 used with momentum factor (default 0.9; set to 0.0 for no momentum optimization)
            a_beta2: Beta2 used with RMSprop factor (default 0.999; set to 0.0 for no RMSprop optimization)
            a_epsilon: A small value Epsilon used to avoid large/unstable values due to denominator -> 0 (default 10E-08)
            
        Returns:
            None: Values are updated in self._config:
                       batch_size, alpha, max_epochs, lambda, m, optim, beta1, beta2, epsilon
        """

        # Confirm that the model is configured before attempting to initialize the parameters
        assert self._config['is_configured'], f"Cannot set the fit configuration until model is configured: is_configured = {self._config['is_configured']} [{type(self._config['is_configured'])}]"

        # Confirm a valid batch size has been provided
        assert int(a_batch_size) > 0, f"Batch size must be a number > 0: a_batch_size = {a_batch_size} [{type(a_batch_size)}]"
        self._config['batch_size'] = int(a_batch_size)   # Batch size to use per iteration
        
        # Confirm a valid learning rate alpha has been provided
        assert float(a_alpha) > 0.0, f"Learning Rate alpha must be a number > 0.0: a_alpha = {a_alpha} [{type(a_alpha)}]"
        self._config['alpha'] = float(a_alpha)       # Learning rate alpha to apply
                
        # Confirm a valid limit on the number of epochs (i.e., passes through the training set) has been provided
        assert int(a_max_epochs) > 0, f"Maximum Epochs (i.e., passes through training set) must be a number > 0: a_max_epochs = {a_max_epochs} [{type(a_max_epochs)}]"
        self._config['max_epochs'] = int(a_max_epochs)   # Maximum Epochs
        
        # Confirm a valid regularization rate lambda has been provided
        assert float(a_lambda) >= 0.0, f"Regularization Rate lambda must be a number >= 0.0: a_lambda = {a_lambda} [{type(a_lambda)}]"
        self._config['lambda'] = float(a_lambda)       # Regularization rate
                
        # Confirm that the feature count is valid (i.e., matches first entry in all_layers)
        assert a_X.shape[0] == self._config['all_layers'][0], f"Number of features in X must match layers specification: X.shape = {a_X.shape}, all_layers[0] = {self._config['all_layers'][0]}"
        
        # Confirm that the output count is valid (i.e., matches last entry in all_layers)
        # NOTE: y should be a 2 dimensional numpy array,
        #       but because there is usually just 1 output (i.e., 1 row)
        #       y *could* be specified with just 1 dimension.
        #       => Try to handle this gracefully...
        if a_y.ndim > 1:
            # y has 2 (or more) dimension, so the 1st dim: # of outputs, 2nd dim: # of examples
            assert a_y.shape[0] == self._config['all_layers'][-1], f"Number of outputs in y must match layers specification: y.shape = {a_y.shape}, all_layers[L={self._config['L']}] = {self._config['all_layers'][-1]}"
            
            # Keep track of the number of examples in a_y for a check below
            n_y_examples = a_y.shape[1]
        
        else:
            # y has only 1 dimension (should be 2!), so *assume* 1st dim: # of examples and # of outputs == 1
            assert self._config['all_layers'][-1] == 1, f"Number of outputs in y must match layers specification: y.shape = {a_y.shape}, all_layers[L={self._config['L']}] = {self._config['all_layers'][-1]}"

            # Keep track of the number of examples in a_y for a check below
            n_y_examples = a_y.shape[0]
                       
        # Confirm X and Y training data are consistent with configuration
        assert a_X.shape[1] == n_y_examples, f"Number of training examples are different for X vs y: X.shape = {a_X.shape}, y.shape = {a_y.shape}"
        self._config['m'] = int(a_X.shape[1])    # Number of Examples
        
        # Confirm a valid optimizer has been selected
        valid_optimizers=['gd', 'adam']
        assert str(a_optim) in valid_optimizers, f"Optimizer must be selected from {valid_optimizers}: a_optim = {a_optim} [{type(a_beta1)}]"
        self._config['optim'] = str(a_optim)    # Optimizer - set to 'none' for no optimization

        # Confirm a valid beta1 value (momentum) has been provided for use with adam optimizer
        assert float(a_beta1) >= 0.0 and float(a_beta1) < 1.0, f"Beta1 (momentum) must be a number >= 0.0 and < 1.0: a_beta1 = {a_beta1} [{type(a_beta1)}]"
        self._config['beta1'] = float(a_beta1)  # Beta1 (momentum) - set to 0.0 for no momentum factor
                
        # Confirm a valid beta2 value (RMSprop) has been provided for use with adam optimizer
        assert float(a_beta2) >= 0.0 and float(a_beta2) < 1.0, f"Beta2 (RMSprop) must be a number >= 0.0 and < 1.0: a_beta2 = {a_beta2} [{type(a_beta2)}]"
        self._config['beta2'] = float(a_beta2)  # Beta2 (RMSprop) - set to 0.0 for no RMSprop factor

        # Confirm a valid beta2 value (RMSprop) has been provided for use with adam optimizer
        assert float(a_epsilon) > 0.0, f"Beta2 (RMSprop) must be a small number > 0.0: a_epsilon = {a_epsilon} [{type(a_epsilon)}]"
        self._config['epsilon'] = float(a_epsilon)    # Epsilon
        

    def _propagate_forward(self, a_X_batch):
        """
        Perform forward propagation calculations using the specified batch examples.
        
        Arguments:
            a_X_batch: A batch subset of input values
            
        Returns:
            A[L]: Predicted output from the output layer
        """

        # DEBUG
        DEBUG_LEVEL = 0
        
        # Confirm that the model is configured before attempting to initialize the parameters
        assert self._config['is_configured'], f"Cannot propagate forward until the model is configured: is_configured = {self._config['is_configured']} [{type(self._config['is_configured'])}]"

        # Get the the output layer number
        L_val = self._config['L']

        # Initialize A0 to X (the input layer)
        self._cache['A']['A0'] = a_X_batch
        
        # Loop through all hidden layers (i.e., 1 through L-1) and the output layer L)
        for layer in range(1, L_val+1):

            if DEBUG_LEVEL >= 2:
                d_text  = f"\nDEBUG: _propagate_forward(): Starting processing for Layer {layer} of {L_val}"
                print(d_text)

            # Ensure that the needed parameter values are present
            assert 'W'+str(layer) in self._param['W'].keys(), f"Missing parameter value W{layer} needed to update layer {layer}: W = {self._param['W']}"
            assert 'b'+str(layer) in self._param['b'].keys(), f"Missing parameter value b{layer} needed to update layer {layer}: b = {self._param['b']}"

            # Ensure that the needed cache values needed to perform the update are present
            assert 'A'+str(layer-1) in self._cache['A'].keys(), f"Missing cache value A{layer-1} needed to update layer {layer}: A cache = {self._cache['A']}"

            if DEBUG_LEVEL >=2:
                d_text  = f"\nDEBUG: _propagate_forward(): After Validity Checks\n"
                d_text += f"W{layer}.shape: {self._param['W']['W'+str(layer)].shape}, "
                d_text += f"A{layer-1}.shape: {self._cache['A']['A'+str(layer-1)].shape}, "
                print(d_text)
            
            # Calculate the linear function
            Z_val = np.dot(self._param['W']['W'+str(layer)], self._cache['A']['A'+str(layer-1)] ) + self._param['b']['b'+str(layer)]
            
            if DEBUG_LEVEL >=2:
                d_text  = f"\nDEBUG: _propagate_forward(): Linear Equation (Z)\n"
                d_text += f"Z{layer}.shape: {Z_val.shape}, "
                print(d_text)
            
            # Calculate the activation
            if (layer < L_val):

                # Use ReLU for all hidden layers 1 through L-1
                A_val = relu( Z_val )
                
            else:
                # Use Sigmoid for the output layer L
                A_val = sigmoid( Z_val )
            
            if DEBUG_LEVEL >=2:
                d_text  = f"\nDEBUG: _propagate_forward(): Linear Activaion (A)\n"
                d_text += f"A{layer}.shape: {A_val.shape}, "
                print(d_text)
            
            # Cache the calculated Z and A values
            self._cache['Z']['Z'+str(layer)] = Z_val
            self._cache['A']['A'+str(layer)] = A_val
                
        # Return the prediction from the output layer A[L]
        return self._cache['A']['A'+str(L_val)]

    
    def _propagate_backward(self, a_y_batch):
        """
        Perform backward propagation calculations using the specified batch examples.
        
        Arguments:
            a_y_batch: A batch subset of actual output values
            
        Returns:
            None: Values are updated for each layer in self._cache: dA, dZ, dW, db           
        """

        # DEBUG
        DEBUG_LEVEL = 0
        
        # Confirm that the model is configured before attempting to initialize the parameters
        assert self._config['is_configured'], f"Cannot propagate forward until the model is configured: is_configured = {self._config['is_configured']} [{type(self._config['is_configured'])}]"

        # Get the the output layer number
        L_val = self._config['L']
        
        # Get the batch size
        batch_size_val = self._config['batch_size']

        
        # Ensure that the needed cache values needed to perform the update are present
        assert 'A'+str(L_val) in self._cache['A'].keys(), f"Missing cache value A{L_val} needed to update layer {layer}: A cache = {self._cache['A']}"

        # Get the predicted output of the output layer: A[L]
        AL_val = self._cache['A']['A'+str(L_val)]

        # Calculate and cache dA[L] and dZ[L], which require only the predicted and actual output values
        self._cache['dA']['dA'+str(L_val)] = - (a_y_batch / AL_val) + (1-a_y_batch) / (1-AL_val)
        self._cache['dZ']['dZ'+str(L_val)] = AL_val - a_y_batch

        # Loop through all hidden and output layers in reverse order:
        # output layer L then hidden layers (i.e., L-1 through 1)
        for layer in reversed(range(1, L_val+1)):

            if DEBUG_LEVEL >= 2:
                d_text  = f"\nDEBUG: _propagate_backward(): Starting processing for Layer {layer} of {L_val}"
                print(d_text)

            # Ensure that the needed parameter values are present
            assert 'W'+str(layer) in self._param['W'].keys(), f"Missing parameter value W{layer} needed to update layer {layer}: W = {self._param['W']}"

            # Retrieve W parameters for (current layer)
            W = self._param['W']['W'+str(layer)]
            
            # Ensure that the needed cache values needed to perform the update are present
            assert 'dZ'+str(layer) in self._cache['dZ'].keys(), f"Missing cache value dZ{layer} needed to update layer {layer}: dZ cache = {self._cache['dZ']}"
            assert 'A'+str(layer-1) in self._cache['A'].keys(), f"Missing cache value A{layer-1} needed to update layer {layer}: A cache = {self._cache['A']}"
                        
            # Retrieve dZ for (current layer) - already cached
            dZ = self._cache['dZ']['dZ'+str(layer)]

            # Retrive A for (current layer minus 1) - already cached
            A_m1 = self._cache['A']['A'+str(layer-1)]

            if DEBUG_LEVEL >=2:
                d_text  = f"\nDEBUG: _propagate_backward(): After Retrieving Cached Values\n"
                d_text += f"W{layer}.shape: {self._param['W']['W'+str(layer)].shape}, "
                d_text += f"dZ{layer}.shape: {self._cache['dZ']['dZ'+str(layer)].shape}, "
                d_text += f"A{layer-1}.shape: {self._cache['A']['A'+str(layer-1)].shape}, "
                print(d_text)
            
            if layer > 1:
                # Ensure that the needed cache values needed to perform the update are present
                assert 'Z'+str(layer-1) in self._cache['Z'].keys(), f"Missing cache value Z{layer-1} needed to update layer {layer}: Z cache = {self._cache['Z']}"

                # Retrive Z for (current layer minus 1) - already cached
                Z_m1 = self._cache['Z']['Z'+str(layer-1)]

                # Calculate dA for (current layer minus 1): dA_m1
                dA_m1 = np.dot(W.T, dZ)
                self._cache['dA']['dA'+str(layer-1)] = dA_m1

                # Calculate dZ for (current layer minus 1): dZ_m1
                dZ_m1 = dA_m1 * relu( Z_m1, a_calc_derivative = True )
                self._cache['dZ']['dZ'+str(layer-1)] = dZ_m1
                
                if DEBUG_LEVEL >=2:
                    d_text  = f"\nDEBUG: _propagate_backward(): Layer is > 1:\n"
                    d_text += f"Z{layer-1}.shape: {self._cache['Z']['Z'+str(layer-1)].shape}, "
                    d_text += f"dA{layer-1}.shape: {self._cache['dA']['dA'+str(layer-1)].shape}, "
                    d_text += f"dZ{layer-1}.shape: {self._cache['dZ']['dZ'+str(layer-1)].shape}, "
                    print(d_text)
            
            # Calculate dW for (current layer): dW
            # NOTE: Need to use the batch size, not total number of examples
            dW = (1/batch_size_val) * np.dot(dZ, A_m1.T)
            self._cache['dW']['dW'+str(layer)] = dW
            
            # Calculate db for (current layer): db
            # NOTE: Need to use the batch size, not total number of examples
            db = (1/batch_size_val) * np.sum(dZ, axis=1, keepdims=True)
            self._cache['db']['db'+str(layer)] = db
            
            if DEBUG_LEVEL >=2:
                d_text  = f"\nDEBUG: _propagate_backward(): After Retrieving Cached Values\n"
                d_text += f"dW{layer}.shape: {self._cache['dW']['dW'+str(layer)].shape}, "
                d_text += f"db{layer}.shape: {self._cache['db']['db'+str(layer)].shape}, "
                print(d_text)

    
    def _update_param(self):
        """
        Update the coefficients W and b based upon propagation already performed 
        
        Arguments:
            None: Needed info is obtained from:
                    * self._config: L, alpha
                    * self._cache: dW, db

        Returns:
            None: Values are updated for each layer in self._param: W, b
        """
             
        # Confirm that the model is configured before attempting to initialize the parameters
        assert self._config['is_configured'], f"Cannot update parameters W and b until the model is configured: is_configured = {self._config['is_configured']} [{type(self._config['is_configured'])}]"

        # Loop through all hidden and the final output layer (i.e., 1 through L)
        for layer in range(1, self._config['L']+1):

            # Ensure that the W and b parameter values are present
            assert 'W'+str(layer) in self._param['W'].keys(), f"Missing parameter value W{layer} needed to update layer {layer}: W = {self._param['W']}"
            assert 'b'+str(layer) in self._param['b'].keys(), f"Missing parameter value b{layer} needed to update layer {layer}: b = {self._param['b']}"

            # Ensure that the dW and db cache values needed to perform the update are present
            assert 'dW'+str(layer) in self._cache['dW'].keys(), f"Missing cache value dW{layer} needed to update W{layer}: dW cache = {self._cache['dW']}"
            assert 'db'+str(layer) in self._cache['db'].keys(), f"Missing cache value db{layer} needed to update b{layer}: db cache = {self._cache['db']}"
                        
            # Adjust each parameter by Learning Rate * Derivative for that layer
            self._param['W']['W'+str(layer)] -= self._config['alpha'] * self._cache['dW']['dW'+str(layer)]
            self._param['b']['b'+str(layer)] -= self._config['alpha'] * self._cache['db']['db'+str(layer)]

    
    def _calculate_cost(self, a_y_batch):
        """
        Calculate the cost (loss) and actual vs. prediction deviations for one batch
        
        Arguments:
            a_y_batch: A batch subset of actual output values

        Returns:
            cost_val: Cost(Loss) for this batch -- not scaled by number of examples or batch size
            deviations_val: Sum of actual vs. prediction deviations -- not scaled by number of examples or batch size
        """
        
        #DEBUG
        DEBUG_LEVEL = 0
        
        # Confirm that the model is configured before attempting to initialize the parameters
        assert self._config['is_configured'], f"Cannot update parameters W and b until the model is configured: is_configured = {self._config['is_configured']} [{type(self._config['is_configured'])}]"

        # Get the the output layer number
        L_val = self._config['L']
        
        # Get the batch size
        n_batch_size = self._config['batch_size']
        
        # Get the regularization rate
        lambda_reg_rate = self._config['lambda']

        # Get the predicted output of the output layer (i.e., A[L])
        AL_val = self._cache['A']['A'+str(L_val)]

        # Calculate the cost for this iteration
        # cost_val = - (1/n_batch_size) * np.sum( a_y_batch*np.log(AL_val) + (1-a_y_batch)*np.log(1-AL_val) )
        # Revised: Calculate only the total cost here,
        # and then will normalize by the number of examples in the calling function
        cost_val = - np.sum( a_y_batch*np.log(AL_val) + (1-a_y_batch)*np.log(1-AL_val) )
        
        # Calculate the regularization cost for this iteration if required
        if lambda_reg_rate > 0.0:
            # Loop through each of the layers
            cost_reg = 0

            for layer in range(1, L_val+1):

                if DEBUG_LEVEL >= 2:
                    d_text  = f"\nDEBUG: _calculate_cost(): Calculating regularization cost for Layer {layer} of {L_val}"
                    print(d_text)

                # Ensure that the needed parameter values are present
                assert 'W'+str(layer) in self._param['W'].keys(), f"Missing parameter value W{layer} needed to update layer {layer}: W = {self._param['W']}"

                if DEBUG_LEVEL >=2:
                    d_text  = f"\nDEBUG: _calculate_cost(): After Validity Checks\n"
                    d_text += f"W{layer}.shape: {self._param['W']['W'+str(layer)].shape}, "
                    print(d_text)

                # Calculate and aggregate the sum of squares of the W coefficient for this layer
                cost_reg += np.sum( np.square( self._param['W']['W'+str(layer)] ) )

            # Scale the regularization cost by 1/(number of samples) and a_lambda/2.0
            # cost_reg *= 1.0/n_batch_size * lambda_reg_rate/2.0
            # Revised: Calculate only the total regularization cost here,
            # and then normalize by the number of samples in the calling function
            cost_reg *= lambda_reg_rate/2.0
            
            # Add this regularization cost to the overall cost
            cost_val += cost_reg
            
        # Calculate the accuracy for this batch
        # NOTE: A-Y = dZ, which is already calculated and cached,
        #       but will just use A and Y directly here to remove dependency on back propagation
        # accuracy_val = 1.0 - ( np.sum(np.round(np.abs(AL_val-a_y_batch))) / n_batch_size )
        # Revised: Calculate only the total regularization cost here,
        # and then normalize by the number of samples in the calling function
        deviations_val = np.sum( np.round(np.abs(AL_val-a_y_batch)) )

        return cost_val, deviations_val


    def _update_hist(self, a_cost, a_accuracy ):
        """
        Add one entry to the fit history
        
        Arguments:
            a_cost: Cost(Loss) calculated for one epoch
            a_accuracy: Batch accuracy for one epoch
            
        Returns:
            None: Values are updated in self._hist: cost, accuracy, timestamp
        """

        # Confirm that the model is configured before attempting to initialize the parameters
        assert self._config['is_configured'], f"Cannot update fit history until the model is configured: is_configured = {self._config['is_configured']} [{type(self._config['is_configured'])}]"

        # Add the Cost value J(w,b) to the fit history
        self._hist['cost'].append(np.squeeze(a_cost))

        # Add the accuracy to the fit history
        self._hist['accuracy'].append(np.squeeze(a_accuracy))
        
        # Add a timestamp for when this update was performed
        self._hist['timestamp'].append( datetime.datetime.now() )

            
    # ************************** PUBLIC METHODS **************************
    def configure(self, a_layers = None):
        """
        Configure the number and size of model layers
        
        Arguments:
            a_layers: A list providing the number of nodes/units in each layer:
                        Layer 0: Number of features in input X
                        Layers 1 through L-1: Number of nodes/units in hidden layers
                        Layer L: Number of outputs in Y
                        
        Returns:
            None: Values are updated in:
                    * self._config: all_layers, L, n_x, n_y, is_configured
                    * self._param: W, b
        """
        
        # Confirm that a valid number of hidden layers has been provided
        assert isinstance( list(a_layers), list), f"Must provide a list of node/unit counts for input, hidden, and output layers: layers = {a_layers} [{type(a_layers)}]"
        assert len(list(a_layers)) > 1, f"Must provide a list of node/unit counts for all input, hidden, and output layers: layers = {a_layers} [{type(a_layers)}]"
        assert min(list(a_layers)) > 0, f"Each layer must have > 0 nodes/units: layers = {a_layers} [{type(a_layers)}]"
        self._config['all_layers'] = list(a_layers)             # List of all input, hidden, and output layer unit/node counts
        self._config['L'] = len(list(a_layers))-1               # Number of hidden + output layers (not incl input layer)
        self._config['n_x'] = int(a_layers[0])                  # Number of Input Features (X)
        self._config['n_y'] = int(a_layers[self._config['L']])  # Number of Outputs (y)
                
        # Confirmed that the configuration has been set
        self._config['is_configured'] = True;
        
        # Now that configuration is set,
        # initialize the parameters W and b using the configuration information
        self._init_param()

    def _random_mini_batches(self, a_X_train, a_y_train, a_batch_size = 32):
        """
        Create mini-batches of randomly selected examples from the training set
        
        Arguments:
            a_X: Input features (numpy array) ~ shape( # features, # examples )
            a_y: Output labels (numpy array) ~ shape ( # outputs, # examples )
            a_batch_size: Number of examples to use for validation for each iteration

        Returns: A list of minibatches, where each list element is a tuple (mb_X, mb_y)
                 * NOTE: The last minibatch may be smaller than the others if the
                         total number of examples is not evenly divisible by the specified mini-batch size
                         
        """
        
        # Determine the number of examples in the training set
        m_val=a_X_train.shape[1]

        # Build a list of random index values of size m_val
        random_index_list = list(np.random.permutation(m_val))

        # Shuffle the training examples and labels using the randomized list of indices
        shuffled_X = a_X_train[:,random_index_list]
        shuffled_y = a_y_train[:,random_index_list].reshape(1,-1)

        # Calculate the number of complete mini-batches
        # given the # of training examples and the batch_size using integer division
        n_complete_batches = m_val // a_batch_size

        # Populate the minibatches in a list
        mb_list = []
        for k in range(0, n_complete_batches):
            mb_X = shuffled_X[:,k*a_batch_size:(k+1)*a_batch_size]
            mb_y = shuffled_y[:,k*a_batch_size:(k+1)*a_batch_size]
            mb_list.append( (mb_X, mb_y) )

        # If the # training examples are not evenly divisible by the batch size,
        # calculate the size of the last mini-batch
        if m_val % a_batch_size != 0:
            last_batch_size = m_val - n_complete_batches*batch_size_val
            mb_X = shuffled_X[:,-last_batch_size:]
            mb_y = shuffled_y[:,-last_batch_size:]
            mb_list.append( (mb_X, mb_y) )

        # Return the list of mini-batches
        return mb_list
         

    def fit(self, a_X_train, a_y_train, a_alpha = 0.01, a_batch_size = 32, a_max_epochs = 10000, a_lambda = 0.0, a_optim='gd', a_beta1=0.9, a_beta2=0.999, a_epsilon=10e-08):
        """
        Fit the model coefficients w and b to the training data with specified arguments
        
        Arguments:
            a_X: Input features (numpy array) ~ shape( # features, # examples )
            a_y: Output labels (numpy array) ~ shape ( # outputs, # examples )
            a_alpha: Learning Rate
            a_batch_size: Number of examples to use for validation for each iteration
            a_max_epochs: Maximum number of epochs (i.e., full passes through the training set) to allow
            a_lambda: Regularization Rate (default 0.0 => No regularization)
            a_optim: Optimizer - 'none' or 'adam'
            a_beta1: Beta1 used with momentum factor (default 0.9; set to 0.0 for no momentum optimization)
            a_beta2: Beta2 used with RMSprop factor (default 0.999; set to 0.0 for no RMSprop optimization)
            a_epsilon: A small value Epsilon used to avoid large/unstable values due to denominator -> 0 (default 10e-08)

        Returns:
            None: Values are updated in:
                    * self._param: W, b
                    * self._cache: Z, A, dA, dZ, dW, db, v, v_corrected, s, s_corrected
                    * self._hist:  cost, accuracy
        """
        
        # Set DEBUG_LEVEL
        DEBUG_LEVEL = 1
        
        # Confirm that the model is configured before attempting to initialize the parameters
        assert self._config['is_configured'], "Cannot fit the model to training data until the model is configured"

        # Set the fit configuration, including basic error checking
        self._set_fit_config(a_X_train, a_y_train, a_alpha, a_batch_size, a_max_epochs, a_lambda, a_optim, a_beta1, a_beta2, a_epsilon)
        # Get needed values from config info
        max_epochs_val = self._config['max_epochs']
        m_val = self._config['m']
        batch_size_val = self._config['batch_size']
        
        # Set the reporting interval at a power of 10 based upon max_epochs_val
        report_interval = np.power(10,np.round(np.log10(max_epochs_val),0)-1)

        # Display a progress update
        if DEBUG_LEVEL >=2:
            d_text  = f"\nDEBUG: fit(): After Calculation of # of Mini-Batches\n"
            d_text += f"max_epochs (# of passes through training set): {max_epochs_val}\n"
            d_text += f"m (# of training examples): {m_val}, "
            d_text += f"batch_size: {batch_size_val}"            
            print(d_text)    

        # Initialize fit history
        self._hist['cost'] = []              # List of cost/loss per epoch
        self._hist['accuracy'] = []          # List of training accuracy per epoch
        
        # Loop through epochs until max_epochs_val is reached.
        # For each epoch:
        #     * Create mini-batches of size 'batch_size', with samples randomly shuffled amongst the batches
        #     * Loop through each of the mini-batches to train the model
        #     * Keep a history of the cost and training set accuracy for each epoch
        
        if (DEBUG_LEVEL >= 1):
            t_stamp=datetime.datetime.now().strftime("%m/%d/%y %I:%M:%S %p")
            d_text = f"[{t_stamp}] Starting Model Fitting"
            print(d_text)

        for k in range(max_epochs_val):
                        
            # Create a list of randomly shuffled mini-batches
            mb_list = self._random_mini_batches(a_X_train, a_y_train, batch_size_val)
            n_batches = len(mb_list)
            
            # Display a progress update
            if DEBUG_LEVEL >=2:
                d_text  = f"\nDEBUG: fit(): After Creation of Random Mini-Batches\n"
                d_text += f"# of mini-batches: {n_batches}"            
                print(d_text)
            
            # Initial the per-epoch cost
            cost_total = 0
            deviations_total = 0
            
            # Loop through each mini-batch
            for mb in mb_list:            
                # Retrieve X and y from this minibatch
                X_batch, y_batch = mb

                if DEBUG_LEVEL >=3:
                    d_text  = f"\nDEBUG: fit(): In Mini-Batch Loop\n"
                    d_text += f"# of examples X_batch: {X_batch.shape[1]}, X_batch.shape: {X_batch.shape}, "
                    d_text += f"# of examples y_batch: {y_batch.shape[1]}, y_batch.shape: {y_batch.shape}"
                    print(d_text)

                # Propagate forward to calculate for each layer: Z, A
                self._propagate_forward(X_batch)

                # Propagate backward to calculate for each layer: dA, dZ, dW, db
                self._propagate_backward(y_batch)

                # Update the parameters for each layer: W, b
                self._update_param()

                # Calculate the cost and actual vs. prediction deviations (without 1/m factor)
                cost_val, deviations_val = self._calculate_cost(y_batch)
                
                # Accumulate the cost and accuracy across all mini-batches for the entire epoch
                cost_total += cost_val
                deviations_total += deviations_val

            # Scale the cost by number of examples in the epoch
            cost_epoch = cost_total / m_val
            
            # Scale the cost by number of examples in the epoch
            accuracy_epoch = 1 - (deviations_total / m_val)

            # Update the fit history with the cost and accuracy for this iteration
            self._update_hist(cost_epoch, accuracy_epoch)
        
            # Display a progress update periodically (report_interval is a power of 10)
            if (DEBUG_LEVEL >= 1) and (k % report_interval == 0):
                t_stamp=datetime.datetime.now().strftime("%m/%d/%y %I:%M:%S %p")
                d_text = f"[{t_stamp}] Epoch: {k} => Cost J(w,b)={cost_epoch:0.4f}, Accuracy={accuracy_epoch:0.4f}"
                print(d_text)
                
        # Set the flag that fit has been performed
        self._config['is_fitted'] = True
        
    def predict(self, a_X_vals):
        """
        Predict the y labels associated with the X feature input
        
        Arguments:
            a_X_vals: Input features (numpy array)
            
        Returns
            A_round: Predicted output (numpy array)
        """
        
        # DEBUG
        DEBUG_LEVEL = 0
        
        # Ensure that the model has already been fitted
        assert self._config['is_fitted'], f"The model must be fitted before making predictions: is_fitted = {self._config['is_fitted']}"

        # Use forward propagation to obtain the prediction for the specified feature input
        # NOTE: Flag that the forward propagation results should *not* be cached
        #       since we're only making a prediction, not training the model
        y_predict = self._propagate_forward(a_X_vals)

        if DEBUG_LEVEL >=2:
            d_text  = f"\nDEBUG: predict(): After Prediction\n"
            d_text += f"a_X_vals - Shape: {a_X_vals.shape}, Type: {type(a_X_vals)},\n"
            d_text += f"y_predict - Shape: {y_predict.shape}, Type: {type(y_predict)},"
            print(d_text)    
        
        # Round the values 
        y_predict_rounded = np.round(y_predict).astype(int)
        
        # Return the prediction
        return y_predict_rounded
    
    
    def get_hist(self):
        """
        Returns the fit history
        
        Arguments:
            None
                        
        Returns:
            Dictionary containing the fit history
        """
    
        # Return the dictionary containing fit history
        return self._hist

    def get_config(self):
        """
        Returns configuration information
        
        Arguments:
            None
                        
        Returns:
            Dictionary containing configuration information
        """
    
        # Return the dictionary containing fit history
        return self._config

    