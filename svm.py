

import pandas as pd 
import sklearn.svm as sv
import helper as hp
@hp.timeit
def fitsvm(data):
    svm=sv.SVC(kernel='rbf',C=20.0,gamma=0.1)
    return svm.fit(data[0],data[1])
csv_data=pd.read_csv("C:/Users/DELL/Desktop/niit/3rd semester/bank_contacts.csv")
train_x,train_y,test_x,test_y,labels=hp.split_data(csv_data,y='credit_application')
classifier=fitsvm((train_x,train_y))
predicted=classifier.predict(test_x)
hp.printModelSummary(test_y,predicted)
print(classifier.support_vectors_)




import pandas as pd
import sklearn.svm as sv
import helper as hp

"""
@ is the decorator to call a function from refered module
"""
@hp.timeit
def fitSVM(data):
    #creating the classifier object
    #svm=sv.SVC(kernel='linear',C=20.0)
    svm=sv.SVC(kernel='rbf',C=20.0, gamma=0.1)
    #fit the model
    return svm.fit(data[0],data[1])

#Explore data
csv_data=pd.read_csv("C:/Users/DELL/Desktop/niit/3rd semester/bank_contacts.csv")

#split data into train and test using split_data() of helper module
train_x, train_y, test_x, test_y,labels=hp.split_data(csv_data,y='credit_application')

#train a model on data
classifier=fitSVM((train_x,train_y))

#evaluate model performance
predicted=classifier.predict(test_x)

#print the performance report
hp.printModelSummary(test_y,predicted)

#print support vector information
print(classifier.support_vectors_)