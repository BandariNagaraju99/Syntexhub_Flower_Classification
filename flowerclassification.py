#!/usr/bin/env python
# coding: utf-8

# In[9]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay


# In[7]:


get_ipython().system('pip install seaborn')


# In[10]:


df = pd.read_csv("iris.csv")
df.head()


# In[11]:


print(df.shape)
print(df.info())
df.describe()
df.isnull().sum()


# In[12]:


sns.pairplot(df, hue='species')
plt.show()


# In[14]:


plt.figure(figsize=(6,4))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()


# In[16]:


X = df.drop("species", axis=1)
y = df["species"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# In[17]:


lr = LogisticRegression(max_iter=200)
lr.fit(X_train, y_train)

y_pred_lr = lr.predict(X_test)
acc_lr = accuracy_score(y_test, y_pred_lr)

print("Logistic Regression Accuracy:", acc_lr)


# In[18]:


dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)

y_pred_dt = dt.predict(X_test)
acc_dt = accuracy_score(y_test, y_pred_dt)

print("Decision Tree Accuracy:", acc_dt)


# In[19]:


print("\nModel Comparison:")
print("Logistic Regression:", acc_lr)
print("Decision Tree:", acc_dt)


# In[20]:


cm_lr = confusion_matrix(y_test, y_pred_lr)

disp = ConfusionMatrixDisplay(confusion_matrix=cm_lr,
                              display_labels=lr.classes_)
disp.plot()
plt.title("Logistic Regression Confusion Matrix")
plt.show()


# In[21]:


cm_dt = confusion_matrix(y_test, y_pred_dt)

disp = ConfusionMatrixDisplay(confusion_matrix=cm_dt,
                              display_labels=dt.classes_)
disp.plot()
plt.title("Decision Tree Confusion Matrix")
plt.show()


# In[22]:


def predict_species():
    print("\nEnter flower measurements:")

    sepal_length = float(input("Sepal Length: "))
    sepal_width = float(input("Sepal Width: "))
    petal_length = float(input("Petal Length: "))
    petal_width = float(input("Petal Width: "))

    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

    prediction = lr.predict(features)

    print("\nPredicted Species:", prediction[0])

# Run function
predict_species()


# In[ ]:




