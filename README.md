# ml-scratch-decision-trees
Decision Tree Algorithm

## **Description**
The following is my from scratch implementation of the Decision Tree algorithm.

### **Dataset**

I tested the performance of my model on three datasets: \
\
    &emsp;1. Breast Cancer Dataset \
    &emsp;2. Iris Dataset \
    &emsp;3. Diabetes Dataset

### **Walkthrough**

**1.** Need the following packages installed: sklearn, numpy, collections, and graphviz.

**2.** Once you made sure all these libraries are installed, evrything is simple, just head to main.py and execute it.

**3.** Since code is modular, main.py can easily: \
\
    &emsp;**i.** Load the three datasets \
    &emsp;**ii.** Split data into train and test sets \
    &emsp;**iii.** Build a decision tree classifier \
    &emsp;**iv.** Fit the decision tree classifier \
    &emsp;**v.** Predict on the test set \
    &emsp;**vi.** Plot the decision tree hierarchical graph.

**4.** In main.py I specify a set of hyperparameters, these can be picked by the user. The main ones worth noting are the minimum samples split and maximum depth. These hyperparameters were chosen through trail & error experimentation on each dataset.

### **Results**

For each dataset I will list the minimum samples split values , maximum depth, and test accuracy score.
In addition I offer a decision tree graph for visualization.

**1.** Breast Cancer Dataset:

- Hyperparameters:
     - Minimum Samples Split = 2
     - Maximum Depth = 5
 
- Numerical Result:
     - Accuracy = 94.74%

- See visualization below:

![alt text](https://github.com/ZainUFarhat/ml-scratch-decision-trees/blob/main/plots/bc/bc_decision_tree.png?raw=true)

**2.** Iris Dataset:

- Hyperparameters:
     - Minimum Samples Split = 2
     - Maximum Depth = 5
 
- Numerical Result:
     - Accuracy = 100.0%

- See visualization below:

![alt text](https://github.com/ZainUFarhat/ml-scratch-decision-trees/blob/main/plots/iris/iris_decision_tree.png?raw=true)

**2.** Diabetes Dataset:

- Hyperparameters:
     - Minimum Samples Split = 5
     - Maximum Depth = 3
 
- Numerical Result:
     - Accuracy = 76.4%

- See visualization below:

![alt text](https://github.com/ZainUFarhat/ml-scratch-decision-trees/blob/main/plots/db/db_decision_tree.png?raw=true)