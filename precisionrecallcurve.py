from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np
import pickle
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
from collections import Counter
list1=[]

with open('/Users/harika/PycharmProjects/sentimentalanalysis/venv1/neuralactual','rb') as fp:
    neuralnetactual = pickle.load(fp)
with open('/Users/harika/PycharmProjects/sentimentalanalysis/venv1/neuralpredictions','rb') as fd:
    neuralnetpredict = pickle.load(fd)

list1=neuralnetactual
Counter(list1)
nbins =4
print Counter(list1)
print Counter(neuralnetpredict)
count1=0
count0=0
for i in range(len(neuralnetactual)):
    if neuralnetactual[i] == 1 and  neuralnetpredict[i] == 1:
        count1 = count1+1

    if neuralnetactual[i] == 0 and neuralnetpredict[i] == 0:
        count0 = count0+1

print count1
print count0
objects=['positive','predictedpostive','negative','predictednegative']
y_pos=np.arange(len(objects))
n_groups = 4
x1= [35196,40013]
x2= [35412, 39974]

fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.1
opacity = 0.8

rects1 = plt.bar(index, x1, bar_width,
                 alpha=opacity,
                 color='g',
                 )

rects2 = plt.bar(index + bar_width, x2, bar_width,
                 alpha=opacity,
                 color='r',
                )


plt.xticks(index,objects)
plt.legend()

plt.tight_layout()
plt.show()