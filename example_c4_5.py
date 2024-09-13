import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import numpy as np
 
data = pd.read_csv(r'C:\Users\kamis\OneDrive\Pulpit\inzynierka\tabela1.csv', encoding='latin1')
 
print(data.columns)
 
data_encoded = data.copy()
 
label_encoders = {}
for column in data_encoded.columns[1:]:
    if data_encoded[column].dtype == 'object':
        le = LabelEncoder()
        data_encoded[column] = le.fit_transform(data_encoded[column])
        label_encoders[column] = le
 
X = data_encoded.drop('spacer(x)', axis=1)  
y = data_encoded['spacer(x)']  
 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
 
clf_c45 = DecisionTreeClassifier(criterion='entropy', splitter='best')
 
clf_c45.fit(X_train, y_train)
 
plt.figure(figsize=(12, 8))
tree.plot_tree(clf_c45, filled=True, feature_names=X.columns, class_names=['nie', 'tak'])
plt.show()
 
accuracy_c45 = clf_c45.score(X_test, y_test)
print(f'Accuracy (C4.5 approximation): {accuracy_c45}')
