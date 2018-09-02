from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import pickle
from sklearn.neural_network import MLPClassifier

with open('/Users/harika/PycharmProjects/sentimentalanalysis/venv1/neuralactualv18','rb') as fp:
    neuralnetactual = pickle.load(fp)
with open('/Users/harika/PycharmProjects/sentimentalanalysis/venv1/neuralscorepredictions','rb') as fd:
    neuralnetpredict = pickle.load(fd)

with open('/Users/harika/PycharmProjects/sentimentalanalysis/svmactual','rb') as fa:
    svmactual = pickle.load(fa)
with open('/Users/harika/PycharmProjects/sentimentalanalysis/svmpredict','rb') as fx:
    svmpredict = pickle.load(fx)

with open('/Users/harika/PycharmProjects/sentimentalanalysis/actualpredictionsRF','rb') as fb:
    randomforestactual = pickle.load(fb)
with open('/Users/harika/PycharmProjects/sentimentalanalysis/RandomForestpredict','rb') as fy:
    randomforestpredict = pickle.load(fy)
falsepositiverate, truepositiverate, threshold = roc_curve(neuralnetactual,neuralnetpredict)
roc_auc = auc(falsepositiverate, truepositiverate)

fpr, tpr, threshold1 = roc_curve(svmactual, svmpredict)
roc_aucsvm = auc(fpr, tpr)

fpr1, tpr2, threshold2 = roc_curve(randomforestactual, randomforestpredict)
roc_aucrandomforest = auc(fpr1, tpr2)

plt.figure()
plt.plot(falsepositiverate, truepositiverate, color='darkgreen',
         lw=2, label='MLP Classifier (area = %0.2f)' % roc_auc)
#plt.plot(fpr, tpr, color='orange',
 #        lw=2, label='Support Vector Machine (area = %0.2f)' % roc_aucsvm)
#plt.plot(fpr1, tpr2, color='deeppink',
#         lw=2, label='Random Forest Classifier (area = %0.2f)' % roc_aucrandomforest)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('MLP classifier')
plt.legend(loc="lower right")
plt.show()