import numpy as np 
from helper import *

'''
Homework2: Perceptron Classifier
Sukriti Agarwal
CECS 456
'''

def sign(x):
	return 1 if x > 0 else -1

#-------------- Implement your code Below -------------#

def show_images(data): 
    firstImage = data[0]
    secondImage = data[1]
    plt.imshow(firstImage)
    plt.savefig('5 Plot')
    plt.show()
    plt.imshow(secondImage)
    plt.savefig('1 Plot')
    plt.show()

def show_features(data, label):
    oneX = []
    oneY = []
    fiveX = []
    fiveY = []
    for x in range(len(data)):
        if label[x] == 1:
            oneX.append(data[x][0])
            oneY.append(data[x][1])
        else:
            fiveX.append(data[x][0])
            fiveY.append(data[x][1])
    plt.scatter(oneX,oneY,marker='*',color='red')
    plt.scatter(fiveX,fiveY,marker='+',color='blue')
    plt.savefig('Feature Scatter - Accuracy Test')
    plt.show()

def perceptron(data, label, max_iter, learning_rate):
    w = np.zeros((1,3))
    for i in range(len(data)): 
        s = sum(np.dot(data[i],np.transpose(w)))
        h = sign(s)
        if(label[i]!=h):
            w = w + data[i] * label[i] * learning_rate
    return w

def show_result(data, label, w):
    for i in range(len(data)):
        if(label[i] == 1):
            plt.scatter(data[i][0],data[i][1],marker='*',c='red')
        else:
            plt.scatter(data[i][0],data[i][1],marker='+',c='blue')
    plt.xlabel('Symmetry')
    plt.ylabel('Average Intensity')
    weight = w[0]
    x = np.linspace(np.amin(data[:,:1]),np.amax(data[:,:1]))
    slope = -(weight[0]/weight[2])/(weight[0]/weight[1])
    intercept = -weight[0]/weight[2]
    y = [(slope*i)+intercept for i in x]
    plt.plot(x, y, c="black")
    plt.savefig("Result Scatter - Result Summary")
    plt.show()

#-------------- Implement your code above ------------#
def accuracy_perceptron(data, label, w):
	n, _ = data.shape
	mistakes = 0
	for i in range(n):
		if sign(np.dot(data[i,:],np.transpose(w))) != label[i]:
			mistakes += 1
	return (n-mistakes)/n


def test_perceptron(max_iter, learning_rate):
	#get data
	traindataloc,testdataloc = "../data/train.txt", "../data/test.txt"
	train_data,train_label = load_features(traindataloc)
	test_data, test_label = load_features(testdataloc)
	#train perceptron
	w = perceptron(train_data, train_label, max_iter, learning_rate)
	train_acc = accuracy_perceptron(train_data, train_label, w)	
	#test perceptron model
	test_acc = accuracy_perceptron(test_data, test_label, w)
	return w, train_acc, test_acc


