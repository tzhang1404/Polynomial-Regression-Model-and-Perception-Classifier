import numpy as np
try:
    import matplotlib.pyplot as plt
except:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

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


class Perceptron():
    def __init__(self, max_iterations=200):
        """
        This implements a linear perceptron for classification. A single
        layer perceptron is an algorithm for supervised learning of a binary
        classifier. The idea is to draw a linear line in the space that separates
        the points in the space into two partitions. Points on one side of the 
        line are one class and points on the other side are the other class.
       
        begin initialize weights
            while not converged or not exceeded max_iterations
                for each example in features
                    if example is misclassified using weights
                    then weights = weights + example * label_for_example
            return weights
        end
        
        Note that label_for_example is either -1 or 1.

        Use only numpy to implement this algorithm. 

        Args:
            max_iterations (int): the perceptron learning algorithm stops after 
            this many iterations if it has not converged.

        """
        self.max_iterations = max_iterations
        self.weights = []


    def initializeWeights(self, features):
        self.weights = np.random.uniform(-1,1,features.shape[1])

    def accuracy(self, f0):

        result= (f0[0] * self.weights[0]) + (f0[1] * self.weights[1]) + (f0[2] * self.weights[2])
        if result > 0:
                    result = 1
        else:
                    result = -1
        return result

    def fit(self, features, targets):
        """
        Fit a single layer perceptron to features to classify the targets, which
        are classes (-1 or 1). This function should terminate either after
        convergence (dividing line does not change between interations) or after
        max_iterations (defaults to 200) iterations are done. Here is pseudocode for 
        the perceptron learning algorithm:

        Args:
            features (np.ndarray): 2D array containing inputs.
            targets (np.ndarray): 1D array containing binary targets.
        Returns:
            None (saves model and training data internally)
        """
        newFeatures = np.insert(features, 0, 1, axis = 1)
        self.initializeWeights(newFeatures)


        for epoch in range(self.max_iterations):
            for i in range(newFeatures.shape[0]):
                prediction = self.accuracy(newFeatures[i])
                if prediction != targets[i]:
                    #print("prediction: ", prediction)
                    error = newFeatures[i] * targets[i]
                    #print("error: ", error)
                    self.weights += error
                    




    def predict(self, features):
        """
        Given features, a 2D numpy array, use the trained model to predict target 
        classes. Call this after calling fit.

        Args:
            features (np.ndarray): 2D array containing real-valued inputs.
        Returns:
            predictions (np.ndarray): Output of saved model on features.
        """
        features = np.insert(features, 0, 1, axis = 1)
        predictions = np.zeros(shape = features.shape[0])

        for i in range(features.shape[0]):
            result = (features[i][0] * self.weights[0]) + (features[i][1] * self.weights[1]) + (features[i][2] * self.weights[2])
            if result > 0:
                predictions[i] = 1
            else:
                predictions[i] = -1
        print("predictions: ", predictions)
        return predictions

    def visualize(self, features, targets):
        """
        This function should produce a single plot containing a scatter plot of the
        features and the targets, and the perceptron fit by the model should be
        graphed on top of the points.

        DO NOT USE plt.show() IN THIS FUNCTION.

        Args:
            features (np.ndarray): 2D array containing real-valued inputs.
            targets (np.ndarray): 1D array containing binary targets.
        Returns:
            None (plots to the active figure)
        """
        plt.scatter(features, targets)
        plt.legend(loc='best')
        plt.savefig()
