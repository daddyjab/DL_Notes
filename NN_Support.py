import numpy as np

def sigmoid(a_vector, a_calc_derivative=False):
    """
    Function to calculate the sigmoid activation function
    
    Arguments:
        a_vector: numpy array of values
        a_calc_derivative [OPTIONAL]:
            False: Calculate the sigmoid function (DEFAULT)
            True: Calculate the derivative of the sigmoid function
            
    Returns:
        (See a_calc_derivative above)
    """

    if a_calc_derivative:
        return sigmoid( a_vector, False) * (1 - sigmoid(a_vector, False) )
    
    else:
        return 1 / (1 + np.exp(-a_vector) )

def relu(a_vector, a_calc_derivative=False):
    """
    Function to calculate the Rectified Linear Unit (ReLU) activation function
    
    Arguments:
        a_vector: numpy array of values
        a_calc_derivative [OPTIONAL]:
            False: Calculate the ReLU function (DEFAULT)
            True: Calculate the derivative of the ReLU function
            
    Returns:
        (See a_calc_derivative above)
    """

    if a_calc_derivative:
        return np.where(a_vector > 0, 1, 0)
    
    else:
        return np.maximum(0, a_vector)

def leaky_relu(a_vector, a_calc_derivative=False):
    """
    Function to calculate the Leaky Rectified Linear Unit (ReLU) activation function
    
    Arguments:
        a_vector: numpy array of values
        a_calc_derivative [OPTIONAL]:
            False: Calculate the Leaky ReLU function (DEFAULT)
            True: Calculate the derivative of the Leaky ReLU function
            
    Returns:
        (See a_calc_derivative above)
    """
    leak_factor = 0.01
    if a_calc_derivative:
        return np.where(a_vector > 0, 1, leak_factor)
    
    else:
        return np.maximum(leak_factor*a_vector, a_vector)

def evaluate(a_y_predict, a_y_actual):
    """
    Evaluate the model for accuracy

    Arguments:
        a_y_predict: Predicted output labels
        a_y_actual: Actual output labels

    Returns: A dictionary with the following values:
        'predict_accuracy': Accuracy of the prediction
        'predict_proba_label_1': Probability of a correct prediction when actual label is 1
        'predict_proba_label_0': Probability of a correct prediction when actual label is 0

    Notes:
        a_y_predict and a_y_actual must have the same number of elements when flattened
    """

    y_pred = a_y_predict.reshape(-1)
    y_act = a_y_actual.reshape(-1)
    
    # Error message if the arguments do not have the same number of elements
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

    # Calculate the cost
    # cost_val = - ( 1/np.size(y_act) ) * np.sum( y_act*np.log(y_pred) + (1-y_act)*np.log(1-y_pred) )
    
    # Return the accuracy
    retval = {
        # 'cost': cost_val,
        'accuracy': predict_accuracy,
        'proba_label_1': predict_proba_label_1,
        'proba_label_0': predict_proba_label_0
    }
    return retval    

def plot_fit_history(a_fit_hist, a_valid_info = None):
    """
    Plot the fit history, including Loss (i.e., Cost) and Training Accuracy if provided
    """
    
    # Check fit history arguments
    try:
        # Get the cost and accuracy lists from the fit history dictionary
        h_n_iter = a_fit_hist['n_iter']
        h_cost = np.squeeze(a_fit_hist['cost'])
        h_accuracy = np.squeeze(a_fit_hist['accuracy'])

    except Exception as e:
        # Re-raise the exception that got us here,
        # including any error message passed along
        raise
    
    assert len(h_cost) != 0, f"Fit history cost data is missing: Cost[{len(h_cost)}]"
    assert len(h_accuracy) != 0, f"Fit history accuracy data is missing: Accuracy[{len(h_accuracy)}]"
    assert len(h_cost) == len(h_accuracy), f"Fit history accuracy and cost data have different lengths: Accuracy[{len(h_accuracy)}], Cost[{len(h_cost)}]"
    
    # Check optional validation info arguments, if provided
    try:
        # Get the overall cost and accuracy scalar values from the validation info dictionary
        v_cost = None
        v_accuracy = None

        if isinstance(a_valid_info,dict) and a_valid_info is not None:
            if 'cost' in a_valid_info.keys():
                v_cost = float(a_valid_info['cost'])

            if 'accuracy' in a_valid_info.keys():
                v_accuracy = float(a_valid_info['accuracy'])

    except Exception as e:
        # Re-raise the exception that got us here,
        # including any error message passed along
        raise
    
    fig1 = plt.figure(figsize=(10,10))

    # Create a single plot of all results
    ax1 = fig1.add_subplot( 2,1,1 )
    ax2 = fig1.add_subplot( 2,1,2 )

    # X-axis - number of epochs in the fit history
    x_vals = range(len(h_cost))

    # Plots
    ax1.set_ylim(ymin=0.9*min(h_accuracy), ymax=1.1)
    ax1.plot( x_vals, h_accuracy,
              label='Accuracy (Training)', c='k', linestyle='-')

    ax2.plot( x_vals, h_cost,
              label='Loss (Training)', c='k', linestyle='-')

    # Add text note on the max and min accuracy and loss points
    acc_min_idx = np.argmin(h_accuracy)
    ax1.text( x=acc_min_idx, y=h_accuracy[acc_min_idx]*1.02, c='b',
              s=f"Min: {h_accuracy[acc_min_idx]:.4f}\nEpoch: {acc_min_idx}" )

    acc_max_idx = np.argmax(h_accuracy)
    ax1.text( x=acc_max_idx, y=h_accuracy[acc_max_idx]*1.02, c='b',
              s=f"Max: {h_accuracy[acc_max_idx]:.4f}\nEpoch: {acc_max_idx}" )

    loss_min_idx = np.argmin(h_cost)
    ax2.text( x=loss_min_idx, y=h_cost[loss_min_idx]*1.02, c='r',
              s=f"Min: {h_cost[loss_min_idx]:.4f}\nEpoch: {loss_min_idx}" )

    loss_max_idx = np.argmax(h_cost)
    ax2.text( x=loss_max_idx, y=h_cost[loss_max_idx]*1.02, c='b',
              s=f"Max: {h_cost[loss_max_idx]:.4f}\nEpoch: {loss_max_idx}" )
    
    # Add text note on the accuracy plot at the point when the loss is minimized
    ax1.text( x=loss_min_idx, y=h_accuracy[loss_min_idx]*1.02, c='r',
              s=f"Accuracy: {h_accuracy[loss_min_idx]:.4f}\n@ Min Loss:{h_cost[loss_min_idx]:.4f}\nEpoch: {loss_min_idx}" )

    # If populated, plot the accuracy from the test samples
    try:
        if v_accuracy is not None:
            # ax1.text( x=min(x_vals), y=v_accuracy*1.02, c='darkgreen', s=f"{v_accuracy:.4f}")
            ax1.text( x=max(x_vals), y=v_accuracy*1.02, c='darkgreen', s=f"Testing:\n{v_accuracy:.4f}")
            ax1.hlines(y=v_accuracy, xmin=min(x_vals), xmax=max(x_vals),
                      label='Accuracy (Testing)', color='g', linewidth=3, linestyle=':')    
    except:
        pass

    # If populated, plot the Loss (Cost) from the test samples
    try:
        if v_cost is not None:
            ax2.text( x=min(x_vals), y=v_cost*1.02, c='darkgreen', s=f"{v_cost:.4f}")
            ax2.hlines(y=v_cost, xmin=min(x_vals), xmax=max(x_vals),
                      label='Loss (Testing)', color='g', linewidth=3, linestyle=':')
    except:
        pass

    ax1.legend(loc='lower right')
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Batch Accuracy")
    ax1.set_title("Model Fitting History - Accuracy")

    ax2.legend(loc='upper right')
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Batch Loss")
    ax2.set_title("Model Fitting History - Loss")

    fig1.savefig("docs/DL-xx-Figure-x-Model_Fit_History.png")

