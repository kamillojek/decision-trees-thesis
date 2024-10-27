import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import matplotlib.pyplot as plt

data = pd.read_csv(r'C:\Users\kamis\OneDrive\Pulpit\inzynierka\modified_cars.csv')

X = data.drop('class', axis=1) 
y = data['class'] 

X = pd.get_dummies(X)

class_names = y.unique().astype(str)

accuracies = []
depths = []
nodes = []
leaves = []

for i in range(30):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=None)

    clf_entropy = DecisionTreeClassifier(criterion='entropy')
    clf_entropy.fit(X_train, y_train)
    
    accuracy_entropy = clf_entropy.score(X_test, y_test)
    accuracies.append(accuracy_entropy)
    
    depth_entropy = clf_entropy.tree_.max_depth
    depths.append(depth_entropy)
    
    node_count_entropy = clf_entropy.tree_.node_count
    nodes.append(node_count_entropy)
    
    leaf_count_entropy = clf_entropy.get_n_leaves()
    leaves.append(leaf_count_entropy)
    
    print(f"Powtórzenie {i+1}:")
    print(f"  Dokładność: {accuracy_entropy}")
    print(f"  Głębokość drzewa: {depth_entropy}")
    print(f"  Liczba węzłów: {node_count_entropy}")
    print(f"  Liczba liści: {leaf_count_entropy}\n")

mean_accuracy = np.mean(accuracies)
std_accuracy = np.std(accuracies)
mean_depth = np.mean(depths)
mean_nodes = np.mean(nodes)
mean_leaves = np.mean(leaves)

print(f"Średnia dokładność (Entropy): {mean_accuracy}")
print(f"Odchylenie standardowe (Entropy): {std_accuracy}")
print(f"Średnia głębokość drzewa (Entropy): {mean_depth}")
print(f"Średnia liczba węzłów (Entropy): {mean_nodes}")
print(f"Średnia liczba liści (Entropy): {mean_leaves}")

plt.figure(figsize=(20,10))
tree.plot_tree(clf_entropy, filled=True, feature_names=X.columns, class_names=class_names)
plt.show()

fig, ax = plt.subplots(figsize=(20, 10))
tree.plot_tree(clf_entropy, filled=True, feature_names=X.columns, class_names=class_names, ax=ax)
fig.savefig('decision_tree_entropy_once_cars.png')
