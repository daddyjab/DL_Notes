import numpy as np
import matplotlib.pyplot as plt

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
        'n_examples': Number of actual examples a_y_actual
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
        'n_examples': np.size(y_act),
        'accuracy': predict_accuracy,
        'proba_label_1': predict_proba_label_1,
        'proba_label_0': predict_proba_label_0
    }
    return retval    


def plot_fit_history(a_fit_hist, a_valid_info = None, a_filename = None):
    """
    Plot the fit history, including Loss (i.e., Cost) and Training Accuracy if provided
    """
    
    # DEBUG
    DEBUG_LEVEL = 0
    
    # Check fit history arguments
    try:
        # Get the cost and accuracy lists from the fit history dictionary
        h_cost = list(a_fit_hist['cost'])
        h_accuracy = list(a_fit_hist['accuracy'])
        
        # Get the list of timestamps for each epoch (as 'datetime' objects)
        h_timestamp = list(a_fit_hist['timestamp'])

    except Exception as e:
        # Re-raise the exception that got us here,
        # including any error message passed along
        raise
    
    assert len(h_accuracy) != 0, f"Fit history accuracy data is missing: Accuracy[{len(h_accuracy)}]"
    assert len(h_cost) != 0, f"Fit history cost data is missing: Cost[{len(h_cost)}]"
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
        
    # Check for optional filename for plot figure
    if a_filename is None:
        fig_filename="DL-xx-Figure-x-Model_Fit_History.png"
        
    else:
        fig_filename=str(a_filename)
        
        # If the specified file name has no (valid) extension, append '.png' to the filename
        if max([f_ext in fig_filename for f_ext in [ '.jpeg', '.jpg', '.png' ] ]) == False:
            fig_filename += ".png"
        
    # Determine how many iterations/epochs are in the fit history
    h_n_epochs = len(h_accuracy)

    # Skip some samples if there are too many points to plot
    # NOTE: Scale things so we have no more than 1000 points to plot.
    PLOT_MAX_POINTS = 1000
    point_increment = max(1, h_n_epochs // PLOT_MAX_POINTS)

    if DEBUG_LEVEL >= 2:
        d_text  = f"\nDEBUG: plot_fit_history(): After Argument Checks\n"
        
        d_text += f"h_n_epochs - Value: {h_n_epochs}, Type: {type(h_n_epochs)},\n"
        d_text += f"h_cost - Len: {len(h_cost)}, Type: {type(h_cost)},\n"
        d_text += f"h_accuracy - Len: {len(h_accuracy)}, Type: {type(h_accuracy)},\n"
        
        if DEBUG_LEVEL >= 3:
            d_text += f"h_cost - Value:\n{h_cost},\n"
            d_text += f"h_accuracy - Value:\n{h_accuracy},\n"

        print(d_text)    

    # Figure with 2 subplots: Accuracy and Cost
    fig1 = plt.figure( figsize=(20,15) )

    # Accuracy plot
    ax1 = fig1.add_subplot( 2,1,1 )
    
    # Cost plot
    ax2 = fig1.add_subplot( 2,1,2 )

    # X-axis - number of epochs in the fit history
    x_vals = range(len(h_cost))

    # Plot Accuracy
    ax1.set_ylim(ymin=0.8*min(h_accuracy), ymax=1.1)
    ax1.plot( x_vals[::point_increment], h_accuracy[::point_increment],
              label='Accuracy (Training)', c='k', linestyle='-')

    # Plot Cost
    ax2.set_ylim(ymin=0.8*min(h_cost), ymax=1.1*max(h_cost) )
    ax2.plot( x_vals[::point_increment], h_cost[::point_increment],
              label='Loss (Training)', c='k', linestyle='-')

    # Add text note on points of max and min accuracy
    acc_min_idx = h_accuracy.index(min(h_accuracy))
    acc_min_ts = h_timestamp[acc_min_idx].strftime("%m/%d/%y %I:%M:%S %p")
    ax1.text( x=acc_min_idx, y=h_accuracy[acc_min_idx]*1.02, c='b',
              s=f"Min: {h_accuracy[acc_min_idx]:.4f}\nEpoch: {acc_min_idx} @ {acc_min_ts}" )

    acc_max_idx = h_accuracy.index(max(h_accuracy))
    acc_max_ts = h_timestamp[acc_max_idx].strftime("%m/%d/%y %I:%M:%S %p")
    ax1.text( x=acc_max_idx, y=h_accuracy[acc_max_idx]*1.02, c='b',
              s=f"Max: {h_accuracy[acc_max_idx]:.4f}\nEpoch: {acc_max_idx} @ {acc_max_ts}" )

    # Add text note on points of max and min cost
    loss_min_idx = h_cost.index(min(h_cost))
    loss_min_ts = h_timestamp[loss_min_idx].strftime("%m/%d/%y %I:%M:%S %p")
    ax2.text( x=loss_min_idx, y=h_cost[loss_min_idx]*1.02, c='r',
              s=f"Min: {h_cost[loss_min_idx]:.4f}\nEpoch: {loss_min_idx} @ {loss_min_ts}" )

    loss_max_idx = h_cost.index(max(h_cost))
    loss_max_ts = h_timestamp[loss_max_idx].strftime("%m/%d/%y %I:%M:%S %p")
    ax2.text( x=loss_max_idx, y=h_cost[loss_max_idx]*1.02, c='b',
              s=f"Max: {h_cost[loss_max_idx]:.4f}\nEpoch: {loss_max_idx} @ {loss_max_ts}" )
    
    # Add text note on the accuracy plot at the point when the loss is minimized
    fit_tot_mins = (h_timestamp[-1] - h_timestamp[0]).total_seconds() / 60.0
    fit_min_loss_mins = (h_timestamp[loss_min_idx] - h_timestamp[0]).total_seconds() / 60.0
    s_text  = f"Accuracy: {h_accuracy[loss_min_idx]:.4f}\n@ Min Loss:{h_cost[loss_min_idx]:.4f}\n"
    s_text += f"Epoch: {loss_min_idx}\n"
    s_text += f"Fit Time to Min Loss: {fit_min_loss_mins:.2f} mins.\nTotal Fit Time: {fit_tot_mins:.2f} mins."
    ax1.text( x=loss_min_idx, y=h_accuracy[loss_min_idx]*0.90, c='r', s=s_text )

    # If populated, plot the accuracy from the test samples
    try:
        if v_accuracy is not None:
            # Plot the test sample accuracy on the left
            # ax1.text( x=min(x_vals), y=v_accuracy*1.02, c='darkgreen', s=f"{v_accuracy:.4f}")
            
            # Plot the test sample accuracy on the right
            # ax1.text( x=max(x_vals), y=v_accuracy*1.02, c='darkgreen', s=f"Testing:\n{v_accuracy:.4f}")

            # Plot the test sample accuracy in the center
            ax1.text( x=( max(x_vals)+min(x_vals) )/2.0, y=v_accuracy*1.02, c='darkgreen', s=f"Testing:\n{v_accuracy:.4f}")
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

    # Set Access plot title, axis labeling, and legend position
    ax1.legend(loc='lower right')
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Batch Accuracy")
    ax1.set_title(f"Model Fitting History - Accuracy\nFilename: {fig_filename}")

    # Set Cost plot title, axis labeling, and legend position
    ax2.legend(loc='upper right')
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Batch Loss")
    ax2.set_title(f"Model Fitting History - Loss\nFilename: {fig_filename}")

    # Save the image to file
    fig1.savefig(f"docs/{fig_filename}")
    