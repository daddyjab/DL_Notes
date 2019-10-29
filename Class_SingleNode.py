import numpy as np

class SingleNode():
    """
    class SimpleNode
    
    An implementation of a single neuron node.
    """
    
    def __init__(self):
        """
        Constructor for the SimpleNode class
        """
        
        self.fit_is_done = False

        self.fit_alpha = None
        self.fit_batch_size = None
        self.fit_max_iter = None

        self.fit_n_iter = None
        self.fit_nsamples = None
        self.fit_iterations = None

        self.fit_coefficient_w = None
        self.fit_coefficient_b = None

        self.fit_history = None
                
        self.fit_train_accuracy = None

        self.eval_is_done = False
        self.eval_accuracy = None
        self.eval_predict_prob = None
        
    def __repr__(self):
        """
        Return an "official" string representation of the object
        """
        
        # Get the list of all class attributes
        attribute_list = dict(vars( self ))

        # Remove coefficients from the output if they are present -- too many values to casually display them!
        attribute_list.pop('fit_coefficient_w', None)
        attribute_list.pop('fit_coefficient_b', None)
        attribute_list.pop('fit_history', None)
        
        # Return the list as a string
        retval = f"{self.__class__.__name__}({attribute_list})"        
        return retval
    
    def __str__(self):
        """
        Return an "friendly" string representation of the object
        """
        
        # For now, just return whatever is generated from __repr__
        retval = self.__repr__()
        return retval
        
    def _set_fit_attributes(self, a_X_train, a_y_train, a_alpha, a_batch_size, a_max_iter):
        """
        Perform fit argument checks and set class attributes
        """
        try:
            if int(a_batch_size) > 0:
                self.fit_batch_size = int(a_batch_size)
            else:
                # Create an error message
                err_str  = f"Error in object {self.__class__.__name__}:\n"
                err_str += f"\tBatch size must be a number > 0\n"
                err_str += f"\tbatch_size = {a_batch_size} [{type(a_batch_size)}]"
                raise ValueError(err_str)
                            
        except Exception as e:
            # Re-raise the exception that got us here,
            # including any error message passed along
            raise
            
        # Set attribute: alpha
        try:
            if float(a_alpha) > 0.0:
                self.fit_alpha = float(a_alpha)
            else:
                # Create an error message
                err_str  = f"Error in object {self.__class__.__name__}:\n"
                err_str += f"\tLearning Rate alpha must be a number > 0.0\n"
                err_str += f"\talpha = {a_alpha} [{type(a_alpha)}]"
                raise ValueError(err_str)
                            
        except Exception as e:
            # Re-raise the exception that got us here,
            # including any error message passed along
            raise

        try:
            if (a_max_iter != None):
                if int(a_max_iter) > 0:
                    self.fit_max_iter = int(a_max_iter)
                else:
                    # Create an error message
                    err_str  = f"Error in object {self.__class__.__name__}:\n"
                    err_str += f"\tMaximum Iterations must be a number > 0\n"
                    err_str += f"\tmax_iter = {a_max_iter} [{type(a_max_iter)}]"
                    raise ValueError(err_str)
                            
        except Exception as e:
            # Re-raise the exception that got us here,
            # including any error message passed along
            raise
            
        # Check for errors in the X and y training data
        try:
            # Ensure that the number of x and y examples match
            if a_X_train.shape[0] == a_y_train.shape[0]:
                # Save the number of examples (samples) in the training data
                self.fit_nsamples = a_X_train.shape[0]
                
            # Ensure that at least one feature has been provided
            if a_X_train.shape[1] > 0:
                # Save the number of features in the training data
                self.fit_train_nfeat = a_X_train.shape[1]
                
            else:
                # Create an error message
                err_str  = f"Error in object {self.__class__.__name__}:\n"
                err_str += f"\tNumber of training samples different for X vs y\n"
                err_str += f"\tX_train.shape = {a_X_train.shape}, y_train.shape = {a_y_train.shape}"
                raise ValueError(err_str)
                            
        except Exception as e:
            # Re-raise the exception that got us here,
            # including any error message passed along
            raise
            
    @staticmethod
    def sigmoid(a_vector):
        """
        Static method to calculate the sigmoid activation function
        """
        return 1 / (1 + np.exp(-a_vector) )    

    def fit(self, a_X_train, a_y_train, a_alpha = 0.01, a_batch_size = 32, a_max_iter = None):
        """
        Fit the model coefficients w and b to the training data with specified arguments
        """
        
        # Set DEBUG_LEVEL
        DEBUG_LEVEL = 0

        # Set attributes, including basic error checking
        self._set_fit_attributes(a_X_train, a_y_train, a_alpha, a_batch_size, a_max_iter)
        
        # Do a basic loop through the samples for now
        # NEXT: Create a generator function to automatically manage batches
        
        # Calculate the number of iterations available given
        # the training sample size and the batch size
        n_iter = int(self.fit_nsamples / self.fit_batch_size)
        
        # Limit the number of iterations to fit_max_iter if specified
        if self.fit_max_iter != None:
            n_iter = min(self.fit_max_iter, n_iter)
        
        # Initialize fit history
        self.fit_history = {
            'cost': [],
            'accuracy': []
            }
        
        # Initialize the coefficients w ~shape(self.fit_train_nfeat, 1)
        w = np.zeros(self.fit_train_nfeat).reshape(-1,1)
        
        # Initialize the coefficient b ~shape(1,1)
        b = 0
        
        # Loop through the iterations
        progress_str = ""
        for i in range(n_iter):
            
            # Get the batch of fit_batch_size examples for this iteration for X and y
            # X_batch: a_X_train has shape (examples, features), so transpose to get (features, examples)
            # y_batch: a_y_train has shape (examples, 1), so transpose to get (1, examples)
            # POSSIBILITY: Could implement batching using numpy masked arrays
            # POSSIBILITY: Could implement batching using a generator function
            X_batch = a_X_train[i*self.fit_batch_size:(i+1)*self.fit_batch_size].T
            y_batch = a_y_train[i*self.fit_batch_size:(i+1)*self.fit_batch_size].reshape(-1,1).T
            
            # DEBUG
            if DEBUG_LEVEL >=2:
                print(f"\nDEBUG: fit() -> After Batch")
                print(f"w ~shape: {w.shape}, w.T ~shape: {w.T.shape}")
                print(f"X_batch ~shape: {X_batch.shape}, X_batch.T ~shape: {X_batch.T.shape}")
                print(f"y_batch ~shape{y_batch.shape}, y_batch.T ~shape{y_batch.T.shape}")
            
            # Calculate the linear equation
            Z = np.dot(w.T, X_batch) + b

            if DEBUG_LEVEL >=2:
                print(f"\nDEBUG: fit() -> After Linear Equation (Z)")
                print(f"Z ~shape{Z.shape}, Z.T ~shape{Z.T.shape}")

            # Calculate the output prediction from the sigmoid of the linear function - vectorized: A and Z ~ shape(1,m)
            A = self.sigmoid(Z)

            if DEBUG_LEVEL >=2:
                print(f"\nDEBUG: fit() -> After Activation (A)")
                print(f"A ~shape{A.shape}, A.T ~shape{A.T.shape}")

            # Calculate the partial derivative of Loss by z
            dz = A - y_batch

            if DEBUG_LEVEL >=2:
                print(f"\nDEBUG: fit() -> After Partial Derivative (dz)")
                print(f"dz ~shape{dz.shape}, dz.T ~shape{dz.T.shape}")

            # Calculate the partial derivative of Loss by w
            dw = (1/self.fit_batch_size) * np.dot(X_batch,dz.T)

            if DEBUG_LEVEL >=2:
                print(f"\nDEBUG: fit() -> After Partial Derivative (dw)")
                print(f"dw ~shape{dw.shape}, dw.T ~shape{dw.T.shape}")

            # Calculate the partial derivative of Loss by b
            db = (1/self.fit_batch_size) * np.sum(dz)

            if DEBUG_LEVEL >=2:
                print(f"\nDEBUG: fit() -> After Partial Derivative (db)")
                print(f"db ~shape{db.shape}, db.T ~shape{db.T.shape}")

            # Use the derivatives to adjust the coefficients w and b
            w -= self.fit_alpha * dw
            b -= self.fit_alpha * db
            
            if DEBUG_LEVEL >=2:
                print(f"\nDEBUG: fit() -> After Update of Coefficients (w, b)")
                print(f"w ~shape: {w.shape}, w.T ~shape: {w.T.shape}")

            # Calculate the overall cost (vectorized) - vectorized: A and Y ~ shape(1,m), J ~ scalar
            J = - (1/self.fit_batch_size) * np.sum( y_batch*np.log(A) + (1-y_batch)*np.log(1-A) )

            # Add the Cost value J(w,b) to the fit history
            self.fit_history['cost'].append(J)

            # Calculate the batch accuracy for this iteration
            # NOTE: A-Y = dz, which is already calculated
            # ALSO: dz will be ~0 when A matches Y
            batch_accuracy = 1 - ( np.sum(np.round(np.abs(dz))) / self.fit_batch_size )
            
            # Add the accuracy to the fit history
            self.fit_history['accuracy'].append(batch_accuracy)
            
            # Display a progress update
            if DEBUG_LEVEL >= 1:
                progress_str = f"[{i}]: Batch Cost J(w,b)={J:0.4f}, Batch Accuracy={batch_accuracy:0.4f}"
                print(progress_str)
            
        # Save the coefficient values
        self.fit_coefficient_w = w
        self.fit_coefficient_b = b
        
        # Save the number of iterations performed
        self.fit_iterations = n_iter
        
        # Set the flag that fit has been performed
        self.fit_is_done = True

        # Return a string providing the model attributes
        retval = self.__repr__()   
        return retval
        
    def predict(self, a_X_vals):
        """
        Predict the y labels associated with the X feature input
        """
        
        # Ensure that the model has already been fitted
        assert self.fit_is_done, "The model must be fitted before making predictions"

        # DEBUG
        DEBUG_LEVEL = 0
        
        # DEBUG
        if DEBUG_LEVEL >=2:
            print(f"\nDEBUG: fit() -> After Batch")
            print(f"w ~shape: {self.fit_coefficient_w.shape}, w.T ~shape: {self.fit_coefficient_w.T.shape}")
            print(f"X ~shape: {a_X_vals.shape}, X.T ~shape: {a_X_vals.T.shape}")

        # Calculate the linear equation
        # NOTE: Need the transpose of a_X_vals.T since it is
        #       (examples x features) and we need (features x examples)
        Z = np.dot(self.fit_coefficient_w.T, a_X_vals.T) + self.fit_coefficient_b

        if DEBUG_LEVEL >=2:
            print(f"\nDEBUG: fit() -> After Linear Equation (Z)")
            print(f"Z ~shape{Z.shape}, Z.T ~shape{Z.T.shape}")

        # Calculate the output prediction from the sigmoid of the linear function - vectorized: A and Z ~ shape(1,m)
        A = self.sigmoid(Z)
        
        if DEBUG_LEVEL >=2:
            print(f"\nDEBUG: fit() -> After Activation (A)")
            print(f"A ~shape{A.shape}, A.T ~shape{A.T.shape}")

        # Reshape prediction array A to single dimension
        A = A.reshape(-1)
        
        # Round the values 
        A_round = np.round(A).astype(int)
        
        # Return the prediction
        return A_round

    def evaluate(self, a_y_predict, a_y_actual):
        """
        Evaluate the model for accuracy
        """

        y_pred = a_y_predict.reshape(-1)
        y_act = a_y_actual.reshape(-1)
        
        # Perform basic arg checking 
        assert y_pred.shape == y_act.shape, \
                f"Arrays y_predict [shape {a_y_predict.shape}] and y_actual [shape {a_y_actual.shape}] must have the same dimensions"

        # Calculate the accuracy
        # a_y_predict - a_y_actual will be close to 0 when the prediction is correct,
        # and will be close to +/-1 when the prediction is incorrect
        # So to calculate the overall accuracy:
        # * Take the absolute value of (a_y_predict-a_y_actual)
        # * Round the result so that we get either 1.0 or 0.0
        # * Sum up the result so that we have a count of incorrect predictions
        # * Divide the sum by the number of examples
        # * Subtract the result from 1.0 to get the accuracy of the model
        predict_accuracy = 1.0 - ( np.sum(np.round(np.abs(y_pred-y_act))) / np.size(y_act) )
    
        # Calculate the probability of predicting 1 given label is 1
        # Create a condition array with True where the actual label is 1
        cond_label_is_1 = (y_act == 1)

        # Apply the condition array to the prediction array to get 
        # all prediction values where the actual label is 1
        y_pred_for_label_1 = np.extract(cond_label_is_1, y_pred)
        
        # Summing provides a count of the 1 values
        predict_proba_label_1 = sum(y_pred_for_label_1) / sum(cond_label_is_1)
    
        # Apply the condition array to the prediction array to get 
        # all prediction values where the actual label is 0 (i.e., not 1)
        y_pred_for_label_0 = np.extract(np.logical_not(cond_label_is_1), y_pred)
        
        # Summing provides a count of the 1 values / all of the 0 values
        predict_proba_label_0 = 1 - (sum(y_pred_for_label_0) / sum(np.logical_not(cond_label_is_1)) )

        # Return the accuracy
        retval = {
            'predict_accuracy': predict_accuracy,
            'predict_proba_label_1': predict_proba_label_1,
            'predict_proba_label_0': predict_proba_label_0
        }
        return retval
