
# coding: utf-8

# In[11]:

get_ipython().magic(u'matplotlib inline')
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
digits = load_digits()


# In[19]:

print ("images shape: %s" % str(digits.images.shape))
print ("targets shape: %s" % str(digits.target.shape))


# In[15]:

plt.matshow(digits.images[4], cmap=plt.cm.Greys);


# In[17]:

digits.items


# In[18]:

X= digits.data.reshape(-1, 64)
print (X.shape)


# In[20]:

y = digits.target
print (y.shape)


# In[21]:

print (X)


# In[22]:

print (y)


# In[28]:

from sklearn.decomposition import PCA
pca= PCA(n_components =2)


# In[30]:

pca.fit(X);


# In[31]:

X_pca = pca.transform(X)
X_pca.shape


# In[34]:


plt.scatter(X_pca[:, 0],X_pca[:,1],c=y);


# In[35]:

print (pca.mean_.shape)
print (pca.components_.shape)


# In[39]:

fix, ax = plt.subplots(1, 3)
ax[0].matshow(pca.mean_.reshape(8, 8), cmap=plt.cm.Greys)
ax[1].matshow(pca.components_[0, :].reshape(8, 8), cmap=plt.cm.Greys)
ax[2].matshow(pca.components_[1, :].reshape(8, 8), cmap=plt.cm.Greys);


# In[40]:

from sklearn.manifold import Isomap


# In[41]:

isomap = Isomap(n_components=2,n_neighbors=20)


# In[42]:

isomap.fit(X)


# In[43]:

X_isomap = isomap.transform(X)
X_isomap.shape


# In[46]:

plt.scatter(X_isomap[:, 0], X_isomap[:, 1], c=y)


#     this is a plot of the load_digits dataset after reshaping

# In[47]:

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train,y_test = train_test_split(X, y, random_state =0)


# In[49]:

print (X_train.shape)
print (X_test.shape)
print (y_train.shape)
print (y_test.shape)


# In[72]:

from sklearn.svm import LinearSVC
svm= LinearSVC()
svm.fit(X_train,y_train)
svm.predict(X_train)


# In[51]:

svm.score(X_train, y_train)


# In[54]:

svm.score(X_test, y_test)


# In[59]:

from sklearn.ensemble import RandomForestClassifier
rf= RandomForestClassifier()
rf.fit(X_train, y_train);
rf.score (X_train, y_train)


# In[60]:

rf.score (X_test, y_test)


# In[ ]:

import numpy as np
from sklearn.cross_validation import cross_val_score
scores =  cross_val_score(rf, X_train, y_train, cv=2)
print("scores: %s  mean: %f  std: %f" % (str(scores), np.mean(scores), np.std(scores)))
get_ipython().magic(u'pinfo cross_val_score')


# In[ ]:

rf2 = RandomForestClassifier(n_estimators=10)
scores =  cross_val_score(rf2, X_train, y_train, cv=5)
print("scores: %s  mean: %f  std: %f" % (str(scores), np.mean(scores), np.std(scores)))


# In[70]:

from sklearn.grid_search import GridSearchCV


# In[75]:

param_grid = {'C': 10. ** np.arange(-3, 4)}
grid_search = GridSearchCV(svm, param_grid=param_grid, cv=3, verbose=3)


# In[76]:

grid_search.fit(X_train,y_train)


# In[77]:

print(grid_search.best_params_)
print(grid_search.best_score_)


# In[83]:

get_ipython().magic(u'matplotlib inline')
import matplotlib.pyplot as plt


# In[86]:

plt.plot([c.mean_validation_score for c in grid_search.cv_scores_], label="validation error")
plt.plot([c.mean_training_score for c in grid_search.cv_scores_], label="training error")
plt.xticks(np.arange(6), param_grid['C']); plt.xlabel("C"); plt.ylabel("Accuracy");plt.legend(loc='best');


# In[87]:

plt.plot([c.mean_training_score for c in grid_search.cv_scores_], label="training error")
plt.xticks(np.arange(6), param_grid['C']); plt.xlabel("C"); plt.ylabel("Accuracy");plt.legend(loc='best');


# In[99]:

import pandas as pd
train_data = pd.read_csv("/Users/test/Dropbox/Prashant Kumar/Scikit-learn tutorials/train.csv")
test_data = pd.read_csv("/Users/test/Dropbox/Prashant Kumar/Scikit-learn tutorials/test_with_solutions.csv")
train_data


# In[100]:

y_train = np.array(train_data.Insult)
comments_train = np.array(train_data.Comment)
print (comments_train.shape)
print (y_train.shape)


# In[107]:

print (comments_train[1])
print ("Insult: %d" % y_train[0])


# from sklearn

# In[108]:

from sklearn.feature_extraction.text import CountVectorizer


# In[109]:

cv = CountVectorizer()
cv.fit(comments_train)
print (cv.get_feature_names()[:15])


# In[114]:

print (cv.get_feature_names()[1000:1015])


# In[119]:

X_train = cv.transform(comments_train)
print("X_train.shape: %s" % str(X_train.shape))
print(X_train[0, :])


# In[120]:

from sklearn.svm import LinearSVC
svm= LinearSVC()
svm.fit(X_train,y_train)


# In[121]:

comments_test = np.array(test_data.Comment)
y_test = np.array(test_data.Insult)
X_test = cv.transform(comments_test)
svm.score(X_test, y_test)


# In[127]:

print(comments_test[0])
print("Target: %d, prediction: %d" % (y_test[8], svm.predict(X_test.tocsr()[8])[0]))


# In[ ]:



