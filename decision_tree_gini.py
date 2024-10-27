import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import matplotlib.pyplot as plt

data = pd.read_csv(r'C:\Users\kamis\OneDrive\Pulpit\inzynierka\modified_mushroom.csv')

X = data.drop('edible', axis=1)
y = data['edible']

X = pd.get_dummies(X)

class_names = y.unique().astype(str)

accuracies = []
depths = []
nodes = []
leaves = []

for i in range(30):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=None)

    clf_gini = DecisionTreeClassifier(criterion='gini')
    clf_gini.fit(X_train, y_train)
    
    accuracy_gini = clf_gini.score(X_test, y_test)
    accuracies.append(accuracy_gini)
    
    depth_gini = clf_gini.tree_.max_depth
    depths.append(depth_gini)
    
    node_count_gini = clf_gini.tree_.node_count
    nodes.append(node_count_gini)
    
    leaf_count_gini = clf_gini.get_n_leaves()
    leaves.append(leaf_count_gini)
    
    print(f"Powtórzenie {i+1}:")
    print(f"  Dokładność: {accuracy_gini}")
    print(f"  Głębokość drzewa: {depth_gini}")
    print(f"  Liczba węzłów: {node_count_gini}")
    print(f"  Liczba liści: {leaf_count_gini}\n")

mean_accuracy = np.mean(accuracies)
std_accuracy = np.std(accuracies)
mean_depth = np.mean(depths)
mean_nodes = np.mean(nodes)
mean_leaves = np.mean(leaves)

print(f"Średnia dokładność (Gini): {mean_accuracy}")
print(f"Odchylenie standardowe (Gini): {std_accuracy}")
print(f"Średnia głębokość drzewa (Gini): {mean_depth}")
print(f"Średnia liczba węzłów (Gini): {mean_nodes}")
print(f"Średnia liczba liści (Gini): {mean_leaves}")

plt.figure(figsize=(20,10))
tree.plot_tree(clf_gini, filled=True, feature_names=X.columns, class_names=class_names)
plt.show()

fig, ax = plt.subplots(figsize=(20, 10))
tree.plot_tree(clf_gini, filled=True, feature_names=X.columns, class_names=class_names, ax=ax)
fig.savefig('decision_tree_gini_once.png')