
# coding: utf-8

# In[14]:


import csv
import numpy as np

scores = []
for i in range(10):
    file = open('model_{}_log.csv'.format(i), 'r')
    reader = csv.reader(file)
    row = list(reader)
    print(i, len(row))
    if len(row) != 0:
        scores.append((i, float(row[-11][-1])))

print(scores)    
print(np.argsort(np.array(scores)[:, 1]))


# In[24]:
