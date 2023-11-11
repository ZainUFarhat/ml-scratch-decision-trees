# datasets
from datasets import *

# Decision Tree
from DecisionTrees import *

# utils
from utils import *

# graphviz
from graphviz import Source

# set numpy random seed
np.random.seed(42)

def main():

    """
    Description:
        Trains and tests our decision tree
    
    Parameters:
        None
    
    Returns:
        None
    """

    print('---------------------------------------------------Dataset----------------------------------------------------')
    # dataset hyperparameters
    test_size = 0.2
    random_state = 42
    dataset_name = 'Breast Cancer'
    
    # create an instance of Datasets class
    datasets = Datasets(test_size = test_size, random_state = random_state)

    # load the breast cancer dataset
    feature_names, class_names, X_train, X_test, y_train, y_test = datasets.load_breast_cancer()

    print(f'Loading {dataset_name} Dataset...')
    print(f'\nThe Features of {dataset_name} Dataset are:', ', '.join(feature_names))
    print(f'The Labels of the {dataset_name} Dataset are:', ', '.join(class_names))
    print(f'\n{dataset_name} contains {len(X_train)} train samples and {len(X_test)} test samples.')
    print('---------------------------------------------------Model------------------------------------------------------')
    print('\nDecision Tree Classifier\n')
    print('---------------------------------------------------Training---------------------------------------------------')
    print('Training...\n')

    # decition tree hyperparameters
    min_samples_split = 2
    max_depth = 5

    dtree = DecisionTree(min_samples_split = min_samples_split, max_depth = max_depth)
    dtree.fit(X_train, y_train)

    print('Done Training!') 
    print('---------------------------------------------------Testing----------------------------------------------------')
    print('Testing...\n')
    predictions = dtree.predict(X_test)

    acc = accuracy_fn(y_true = y_test, y_pred = predictions)

    print('{0} Test Accuracy = {1}%'.format(dataset_name, acc))
    print('\nDone Testing!')
    print('---------------------------------------------------Plotting---------------------------------------------------')
    
    title = f'{dataset_name} Decision Tree'
    save_path = 'plots/bc/bc_decision_tree'

    # convert our tree to dot format so we can render it with graphviz
    tree_to_dot_data = dtree.tree_to_dot(feature_names, class_names, title)

    # render the dot data to a png image using graphviz render
    graph = Source(tree_to_dot_data, format = 'png')
    graph.render(filename = save_path, directory = '.', cleanup = True, view = False)
    print('Please refer to plots/bc directory to view decision tree.')
    print('--------------------------------------------------------------------------------------------------------------')

    ######################################################################################################################################

    print('---------------------------------------------------Dataset----------------------------------------------------')
    # dataset hyperparameters
    dataset_name = 'Iris'
    
    # create an instance of Datasets class
    datasets = Datasets(test_size = test_size, random_state = random_state)

    # load the iris dataset
    feature_names, class_names, X_train, X_test, y_train, y_test = datasets.load_iris()

    print(f'Loading {dataset_name} Dataset...')
    print(f'\nThe Features of {dataset_name} Dataset are:', ', '.join(feature_names))
    print(f'The Labels of the {dataset_name} Dataset are:', ', '.join(class_names))
    print(f'\n{dataset_name} contains {len(X_train)} train samples and {len(X_test)} test samples.')
    print('---------------------------------------------------Model------------------------------------------------------')
    print('\nDecision Tree Classifier\n')
    print('---------------------------------------------------Training---------------------------------------------------')
    print('Training...\n')

    # decition tree hyperparameters
    min_samples_split = 2
    max_depth = 5

    dtree = DecisionTree(min_samples_split = min_samples_split, max_depth = max_depth)
    dtree.fit(X_train, y_train)

    print('Done Training!') 
    print('---------------------------------------------------Testing----------------------------------------------------')
    print('Testing...\n')
    predictions = dtree.predict(X_test)

    acc = accuracy_fn(y_true = y_test, y_pred = predictions)

    print('{0} Test Accuracy = {1}%'.format(dataset_name, acc))
    print('\nDone Testing!')
    print('---------------------------------------------------Plotting---------------------------------------------------')
    
    title = f'{dataset_name} Decision Tree'
    save_path = 'plots/iris/iris_decision_tree'

    # convert our tree to dot format so we can render it with graphviz
    tree_to_dot_data = dtree.tree_to_dot(feature_names, class_names, title)

    # render the dot data to a png image using graphviz render
    graph = Source(tree_to_dot_data, format = 'png')
    graph.render(filename = save_path, directory = '.', cleanup = True, view = False)
    print('Please refer to plots/iris directory to view decision tree.')
    print('--------------------------------------------------------------------------------------------------------------')
    #######################################################################################################################################

    print('---------------------------------------------------Dataset----------------------------------------------------')
    # dataset hyperparameters
    dataset_name = 'Diabetes'
    
    # create an instance of Datasets class
    datasets = Datasets(test_size = test_size, random_state = random_state)

    # load the diabetes dataset
    feature_names, class_names, X_train, X_test, y_train, y_test = datasets.load_diabetes()

    print(f'Loading {dataset_name} Dataset...')
    print(f'\nThe Features of {dataset_name} Dataset are:', ', '.join(feature_names))
    print(f'The Labels of the {dataset_name} Dataset are:', ', '.join(class_names))
    print(f'\n{dataset_name} contains {len(X_train)} train samples and {len(X_test)} test samples.')
    print('---------------------------------------------------Model------------------------------------------------------')
    print('\nDecision Tree Classifier\n')
    print('---------------------------------------------------Training---------------------------------------------------')
    print('Training...\n')

    # decition tree hyperparameters 3 2
    min_samples_split = 5
    max_depth = 3

    dtree = DecisionTree(min_samples_split = min_samples_split, max_depth = max_depth)
    dtree.fit(X_train, y_train)

    print('Done Training!') 
    print('---------------------------------------------------Testing----------------------------------------------------')
    print('Testing...\n')
    predictions = dtree.predict(X_test)

    acc = accuracy_fn(y_true = y_test, y_pred = predictions)

    print('{0} Test Accuracy = {1}%'.format(dataset_name, acc))
    print('\nDone Testing!')
    print('---------------------------------------------------Plotting---------------------------------------------------')
    
    title = f'{dataset_name} Decision Tree'
    save_path = 'plots/db/db_decision_tree'

    # convert our tree to dot format so we can render it with graphviz
    tree_to_dot_data = dtree.tree_to_dot(feature_names, class_names, title)

    # render the dot data to a png image using graphviz render
    graph = Source(tree_to_dot_data, format = 'png')
    graph.render(filename = save_path, directory = '.', cleanup = True, view = False)
    print('Please refer to plots/db directory to view decision tree.')
    print('--------------------------------------------------------------------------------------------------------------')


    return None

if __name__ == '__main__':

    # run everything
    main()