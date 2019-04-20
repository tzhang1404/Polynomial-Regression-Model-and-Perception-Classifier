import numpy as np
import math
try:
    import matplotlib.pyplot as plt
except:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
import os


from generate_regression_data import generate_regression_data
from metrics import mean_squared_error
from polynomial_regression import PolynomialRegression
from load_json_data import load_json_data
from perceptron import Perceptron

def transform_data(features):
    """
    Data can be transformed before being put into a linear discriminator. If the data
    is not linearly separable, it can be transformed to a space where the data
    is linearly separable, allowing the perceptron algorithm to work on it. This
    function should implement such a transformation on a specific dataset (NOT 
    in general).

    Args:
        features (np.ndarray): input features
    Returns:
        transformed_features (np.ndarray): features after being transformed by the function
    """
    #calculate the center:
    totalX = 0
    totalY = 0
    for f in range(features.shape[0]):
        x = features[f][0]
        y = features[f][1]
        totalX += x
        totalY += y

    centerX = totalX/features.shape[0]
    centerY = totalY/features.shape[0]

    for f in range(features.shape[0]):
        if calculateDistance(centerX, centerY, features[f]) > 50:
            features[f][1] = 70
        else:
            features[f][1] = -70

    return features


def calculateDistance(centerX, centerY, featureArray):
    x = featureArray[0]
    y = featureArray[1]

    return ((x - centerX) ** 2 + (y - centerY) ** 2) ** 0.5


def splitdata(features, targets, N):
	

    random_indices_training = np.random.choice(features.shape[0], size = N, replace = False)


    train_features = features[random_indices_training]
    train_targets = targets[random_indices_training]

    random_indices_testing = []
    for i in range(features.shape[0]):
        if i not in random_indices_training:
            random_indices_testing.append(i)
    
    test_features = features[random_indices_testing]
    test_targets = targets[random_indices_testing]

    return train_features, train_targets, test_features, test_targets

x, y = generate_regression_data(4, 100, amount_of_noise = 0.1)

def question1(x, y):
	degrees = range(10)
	degreeArray = []
	testing_mseArray = []
	training_mseArray = []
	best_test_predict = []
	best_train_predict = []
	train_features, train_targets, test_features, test_targets = splitdata(x, y, 10)
	for degree in degrees:
		p = PolynomialRegression(degree)
		# print("degree:     " ,degree)
		p.fit(train_features, train_targets)
		predict_targets = p.predict(test_features)
		train_predict_targets = p.predict(train_features)
		testing_mse = mean_squared_error(test_targets, predict_targets)
		training_mse = mean_squared_error(train_targets, train_predict_targets)
		# print("testmse:        ", testing_mse)
		# print("trainmse:       ", training_mse)
		degreeArray.append(degree)
		testing_mseArray.append(math.log(testing_mse))
		training_mseArray.append(math.log(training_mse))
		if degree == 4:
			best_test_predict = predict_targets
		if degree == 9:
			best_train_predict = train_predict_targets


    #part a, taking the log of the mse
	np.asarray(testing_mseArray)
	np.asarray(training_mseArray)
	testing_mseSum = np.sum(testing_mseArray)
	training_mseSum = np.sum(training_mseArray)
	plt.plot(degreeArray, testing_mseArray, 'r', label = 'Testing Error')
	plt.plot(degreeArray, training_mseArray, 'b', label = 'Traning Error')
	plt.axis([0, 10, -35, 20])
	plt.xlabel('Degree')
	plt.ylabel('Log of MSE')
	plt.legend()
	plt.savefig("q1a")

    #part b
	plt.clf()
	plt.plot(train_features, train_targets,'rv', label = 'Training Data')
	plt.plot(test_features, best_test_predict,'b.', label = 'lowest testing error')
	plt.plot(train_features, best_train_predict, 'g.', label = 'lowest training error')
	plt.legend()
	plt.xlabel('Features')
	plt.ylabel('Targets')
	plt.savefig("q1b")



#question1(x, y)
plt.clf()

def question3(features, targets):
	degrees = range(10)
	degreeArray = []
	testing_mseArray = []
	training_mseArray = []
	best_test_predict = []
	best_train_predict = []
	train_features, train_targets, test_features, test_targets = splitdata(x, y, 50)
	for degree in degrees:
		p = PolynomialRegression(degree)
		# print("degree:     " ,degree)
		p.fit(train_features, train_targets)
		predict_targets = p.predict(test_features)
		train_predict_targets = p.predict(train_features)
		testing_mse = mean_squared_error(test_targets, predict_targets)
		training_mse = mean_squared_error(train_targets, train_predict_targets)
		# print("testmse:        ", testing_mse)
		# print("trainmse:       ", training_mse)
		degreeArray.append(degree)
		testing_mseArray.append(math.log(testing_mse))
		training_mseArray.append(math.log(training_mse))
		
		if degree == 4:
			best_test_predict = predict_targets
		if degree == 9:
			best_train_predict = train_predict_targets
		


    #part a, taking the log of the mse
	np.asarray(testing_mseArray)
	np.asarray(training_mseArray)
	testing_mseSum = np.sum(testing_mseArray)
	training_mseSum = np.sum(training_mseArray)
	plt.plot(degreeArray, testing_mseArray, 'r', label = 'Testing Error')
	plt.plot(degreeArray, training_mseArray, 'b', label = 'Traning Error')
	plt.xlabel('Degree')
	plt.ylabel('Log of MSE')
	plt.axis([0, 10, -4, 6])
	plt.legend()
	plt.savefig("q3a")

    #part b
	plt.clf()
	plt.plot(train_features, train_targets, 'rv', label = 'Training Data')
	plt.plot(test_features, best_test_predict, 'b.', label = 'lowest testing error')
	plt.plot(train_features, best_train_predict, 'g.', label = 'lowest training error')
	plt.legend()
	plt.xlabel('Features')
	plt.ylabel('Targets')
	plt.savefig("q3b")


#question3(x, y)



def question7():
	filenames = ['blobs.json', 'circles.json', 'crossing.json', 'parallel_lines.json', 'transform_me.json']

	for name in filenames:
		p = Perceptron()
		features, targets = load_json_data(name)
		if name == 'transform_me.json':
			features = transform_data(features)
		plt.figure(figsize=(6,4))
		plt.scatter(features[:, 0], features[:, 1], c=targets)
		p.fit(features, targets)
		weights = p.weights
		print(weights)
		for i in range(features.shape[0]):
			slope = -(weights[0]/weights[2])/(weights[0]/weights[1])
			intercept = -weights[0]/weights[2]
			y = (slope * features[i][0]) + intercept
			plt.plot(features[i][0], y, 'k_')
		plt.title(name)
		plt.savefig(f'myOutput{name}.png')



question7()








