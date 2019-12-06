import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold as kf


#calculating P(X|Y)
def calc_cond_prob(x, mean, variance):
    return (1/np.sqrt(2*np.pi*variance))*np.exp((-(x-mean)**2)/(2*variance))



#Predict class using Gaussian Naive Bayes
def predict_class_gnb(test_data):
    class0_prob = calc_cond_prob(test_data['f1'], distribution_mean.iloc[0]['f1'], distribution_variance.iloc[0]['f1'])*calc_cond_prob(test_data['f2'], distribution_mean.iloc[0]['f2'], distribution_variance.iloc[0]['f2'])*\
                  calc_cond_prob(test_data['f3'], distribution_mean.iloc[0]['f3'], distribution_variance.iloc[0]['f3'])*calc_cond_prob(test_data['f4'], distribution_mean.iloc[0]['f4'], distribution_variance.iloc[0]['f4'])*\
                  prior[0]

    class1_prob = calc_cond_prob(test_data['f1'], distribution_mean.iloc[1]['f1'], distribution_variance.iloc[1]['f1'])*calc_cond_prob(test_data['f2'], distribution_mean.iloc[1]['f2'], distribution_variance.iloc[1]['f2'])*\
                  calc_cond_prob(test_data['f3'], distribution_mean.iloc[1]['f3'], distribution_variance.iloc[1]['f3'])*calc_cond_prob(test_data['f4'], distribution_mean.iloc[1]['f4'], distribution_variance.iloc[1]['f4'])*\
                  prior[1]

    return 0 if class0_prob>class1_prob else 1


#calculating sigmoid value
def sigmoid(z):
    return 1/(1+exp(-z))


def Gaussian_Naive_Bayes(test_set):
    errors = 0
    for i in range(0,test_set['class_label'].count()):
        class_value = predict_class_gnb(test_set.iloc[i])
        if class_value != test_set.iloc[i]['class_label'] :
            errors += 1

    return (test_set['class_label'].count()-errors)*100 / test_set['class_label'].count()






data = pd.read_csv('bank_data', header = None, names = ['f1','f2','f3','f4','class_label'])
fraction_list = [.01, .02, .05, .1, .625, 1]
kfold = kf(3, True, 1)

gnb_sum_kfold = {}
for i in fraction_list:
    gnb_sum_kfold[i] = 0.0

for tr_ind, te_ind in kfold.split(data):
    for fraction in fraction_list:
        gnb_sum_acc = 0
        for ii in range (0,5):
            train_set = data.iloc[tr_ind].sample(frac=fraction)
            test_set = data.iloc[te_ind]
            frequency = train_set['class_label'].value_counts()

            distribution_mean = train_set.groupby('class_label').mean()
            distribution_variance = train_set.groupby('class_label').var()
            #print(distribution_mean, distribution_variance)
            prior = frequency / train_set['class_label'].count()

            gnb_sum_acc += Gaussian_Naive_Bayes(test_set)


        gnb_avg_frac_acc = gnb_sum_acc/5
        gnb_sum_kfold[fraction] += gnb_avg_frac_acc

gnb_accuracies = []
for key,val in gnb_sum_kfold.items():
    print(val/3)
    gnb_accuracies.append(val/3)

plt.plot(fraction_list, gnb_accuracies, linewidth=2.0)
plt.xlabel("Fractions")
plt.ylabel("Accuracies")
plt.show()
