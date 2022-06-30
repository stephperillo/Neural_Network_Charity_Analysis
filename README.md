# Neural_Network_Charity_Analysis

## Overview
 
The purpose of this analysis is to create a binary classifier that is capable of predicting whether applicants will be successful if funded by Alphabet Soup. The dataset contains organizations that have received funding from Alphabet Soup over the years.  

## Results

#### Deliverable 1: Data Preprocessing

- What variable(s) are considered the target(s) for your model?
        `IS_SUCCESSFUL` is the target for this model.
- What variable(s) are considered to be the features for your model?
  The features for my model are all of the other variables in `application_df`: 
    `APPLICATION_TYPE`,
    `AFFILIATION`,
    `CLASSIFICATION`,
    `USE_CASE`,
    `ORGANIZATION`,
    `STATUS`,
    `INCOME_AMT`,
    `SPECIAL_CONSIDERATIONS`,
    `ASK_AMT`
- What variables are neither targets nor features, and should be removed from the input data?
  `EIN` and `NAME`

  The `APPLICATION_TYPE` and `CLASSIFICATION` variables have more than 10 unique values. Therefore, I binned "rare" categorical values together in a new `Other` column. Then I used Scikit-learn's `OneHotEncoder` to encode categorical variables before splitting the preprocessed data into features and target arrays, and then into training and testing datasets. The last task of preprocessing was standardizing numerical variables using Scikit-Learn’s `StandardScaler` class to scale the data.

#### Deliverable 2: Compile, Train, and Evaluate the Model

Using the preprocessed data, I created a neural network model using Tensorflor Keras.

hidden_nodes_layer1 = 8

hidden_nodes_layer2 = 5

I used the activation function relu for both hidden layers and sigmoid for the output layer. I also created a callback that saves the model's weights every five epochs.

![Deliverable2_accuracy.png](https://github.com/stephperillo/Neural_Network_Charity_Analysis/blob/main/Resources/Deliverable2_accuracy.png)

This model acheived 72.7% accuracy. 

#### Deliverable 3: Optimize the Model

Optimize the model in order to achieve a target predictive accuracy higher than 75% by using any or all of the following:

- Adjusting the input data to ensure that there are no variables or outliers that are causing confusion in the model, such as:
  - Dropping more or fewer columns.
  - Creating more bins for rare occurrences in columns.
  - Increasing or decreasing the number of values for each bin.
- Adding more neurons to a hidden layer.
- Adding more hidden layers.
- Using different activation functions for the hidden layers.
- Adding or reducing the number of epochs to the training regimen.

For the original and all models in this analysis, I kept the number of epochs as 100. 

New Model B Configuration:

hidden_nodes_layer1 = 10

hidden_nodes_layer2 = 8

hidden_nodes_layer3 = 8

![B_accuracy.png](https://github.com/stephperillo/Neural_Network_Charity_Analysis/blob/main/Resources/B_accuracy.png)

This model acheived 72.5% accuracy. For this model, I:
- Increased the number of hidden nodes in layer 1 to 10
- Increased the number of hidden layers to include a third hidden layer
- Changing the activation functions: used tanh for the output layer

New Model C Configuration:

hidden_nodes_layer1 = 50

hidden_nodes_layer2 = 25

hidden_nodes_layer3 = 20

![C_accuracy.png](https://github.com/stephperillo/Neural_Network_Charity_Analysis/blob/main/Resources/C_accuracy.png)

Model C acheived 73.0% accuracy. The updates I made to this model are:
- Increased the number of hidden nodes in layer 1 to 50
- Increased the number of hidden layers to include a third hidden layer and increased the number of nodes in all hidden layers
- Changed the activation functions: used relu and tanh for the hidden layers, and sigmoid for the output layer

New Model D Configuration:

hidden_nodes_layer1 = 75

hidden_nodes_layer2 = 50

hidden_nodes_layer3 = 25

![D_accuracy.png](https://github.com/stephperillo/Neural_Network_Charity_Analysis/blob/main/Resources/D_accuracy.png)

Model D acheived 73.30% accuracy. The updates I made to this model are:
- Removed the `STATUS` column 
- Increased the number of hidden nodes in layer 1 to 75
- Increased the number of hidden layers to include a third hidden layer and increased the number of nodes in all hidden layers
- Changed the activation functions: used relu for all three hidden layers, and sigmoid for the output layer

Even with these modifications, Model D did not achieve 75% accuracy. 

## Summary

These models were not able to achieve 75% accuracy. After these three attempts, the improvements in accuracy were minor and the performance time was not ideal. Each model ran for a few minutes. One can remove more "noisy" data, modify the activation functions further, and increase the number of hidden layers and nodes. These results indicate that these neural networks may not be the best model to solve this classification problem.  

More analysis could be done with a random forest model. The random forest classifier uses averaging to improve the predictive accuracy and control over-fitting [^1]. This dataset is tabluar, so random forest may be a better, faster, and more accurate model to use.  

[^1]: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
