# File: Dataset_Support.py
# Owner: Jeff Brown

# Dependencies: Standard libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math

# Machine Learning - Keras (Tensorflow) -  Dataset Generation
from keras.datasets import mnist      # Images: Handwritten digits 0-9 (28x28 grayscale, 60K train, 10K test)

# Function to import the Keras MNIST handwritten digits sample dataset
def mnist_load_ds():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    return (X_train, y_train), (X_test, y_test)


# Function to plot a list of up to 10 digits on a single subplot
def mnist_plot_digit_list( a_X_list = None, a_y_list = None, a_find_all_digits = False):
    # The first 10 digits from the specified list
    
    # If no list is specified then return None        
    if (a_X_list is None):
        return None
    
    else:
        X_list = list(a_X_list)
        
    if (a_y_list is None):
        return None
    
    else:
        y_list = list(a_y_list)
        
    # Find All Digits flag
    #   If True => Find and plot all digits 0-9 within the list, starting at index 0
    #   If False => Plot up to the first 10 digits in the list
    if a_find_all_digits:

        # Flag is True: Get indices of samples for each of the digits 0-9 within the 1000 sample subset
        # If the digit is not present in the input list then move on to the next digit
        d_i_list = []
        for d in range(10):
            
            try:
                # Add the index at which this digit can be found to the list
                d_i_list.append( y_list.index(d) )
                
            except ValueError:
                # Digit is not present in the input list -- move on to the next digit
                pass

    else:
        # Flag is False: Get the indices for up to the first 10 values in the list
        d_i_list = range( min(10, len(y_list) ))
    
    # The iterpolation method to use for ploting the digit images
    i_type_selected = 'lanczos'

    print("Indices:", d_i_list)

    # Plot Classification Performance results: Best Score vs. Mean Fit Time (ms)
    fig = plt.figure(figsize=(20,9))

    # Create subplots for each of the sampled digits
    for i in range(len(d_i_list)):
        # Create a subplot for this iteration
        ax = fig.add_subplot( math.ceil(len(d_i_list)/min(5, len(d_i_list))), min(5, len(d_i_list)), i+1 )

        # Display a note for each subplot
        point_text = f"Label: {y_list[d_i_list[i]]}"
        point_text += f"\nSample Index: {d_i_list[i]}"
    #     ax.text(1, 2+1.4*point_text.count("\n"), point_text )
        ax.set_title(point_text)

        # Display the image
        ax.imshow(X_list[d_i_list[i]], cmap=plt.cm.Greys, interpolation=i_type_selected)
        
    # Return the number of digits plotted
    return i+1

# Function to plot example images
def plot_examples( a_X_list = None, a_y_list = None, a_find_labels=False, a_label = None):
    """
    Plot examples from the specified list    
    a_X_list:   List of example inputs
    a_y_list:   List of example labels
    a_find_labels:
                False:  Plot all examples
                True:
                        If a_label is None, plot one example of each unique label
                        If a_label not None, plot all examples with the single specified label
    """

    # Parse args    
    X_list = list(a_X_list.squeeze() )
    y_list = list(a_y_list.squeeze() )
        
    # Parse flags to determine which examples to plot
    if a_find_labels == False:
        # Plot all examples
        d_i_list = range( len(y_list) )

    if a_find_labels == True:
        # Plot examples based upon the label

        if a_label is None:
            # Plot examples for each unique label in the input
            label_list = sorted(set(y_list))

            d_i_list = []
            for label in label_list:
                try:
                    # Add the index at which this label can be found to the list
                    d_i_list.append( y_list.index(label) )
                    
                except ValueError:
                    # Digit is not present in the input list
                    # => Shouldn't happen since the list of labels is drawn from the input list
                    pass

        else:
            # Plot all examples with the single specified label 
            d_i_list = []
            for i in range(len(y_list)):
                if y_list[i] == a_label:
                    d_i_list.append(i)
    
    # print("Image Indices to display:", d_i_list)

    # Plot images
    fig = plt.figure(figsize=(20,10))

    # Create subplots for each of the sampled digits
    for i in range(len(d_i_list)):
        # Create a subplot for this iteration
        ax = fig.add_subplot( math.ceil(len(d_i_list)/min(5, len(d_i_list))), min(5, len(d_i_list)), i+1 )

        # Display a note for each subplot
        point_text = f"Label: {y_list[d_i_list[i]]}"
        point_text += f"\nSample Index: {d_i_list[i]}"
    #     ax.text(1, 2+1.4*point_text.count("\n"), point_text )
        ax.set_title(point_text)

        # Display the image
        ax.imshow(X_list[d_i_list[i]], cmap=plt.cm.Greys, interpolation='lanczos')
        
    # Return the number of digits plotted
    return i+1