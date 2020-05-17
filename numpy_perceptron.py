import numpy as np
import matplotlib.pyplot as plt

# define sigmoid function
def sigmoid(dot_product):
    output = 1/(1 + np.exp(-dot_product)) 
    return output

# The setup
np.random.seed(19)
y = np.array([0.11]) # output
X = np.array([2,1,1,4]) # input
w = np.random.rand(4,) # random weights vector
epochs = range(20) # epochs to train 
learning_rate = 0.5 # seems too high
length_of_X = range(len(X)) # for a range
n = len(X) # how many elements are there in X vector?

# plotting management
error_list = []
epochs_x = list(epochs)

for epoch in epochs:
# FEED FORWARD
    print('EPOCH: {}'.format(epoch))
    print('TARGET: {}'.format(y))
    
    # Predicting y
    y_hat = np.array(sigmoid(np.dot(X,w)))
    print('PREDICTION: {}'.format(y_hat))
    # Print weights
    #print('WEIGHTS: {}'.format(w))
    # How wrong is the prediction? 
    #It's not the mean squared error because we don't have multiple outputs
    se = np.power(np.subtract(y,y_hat),2)
    
    print('ERROR: {}'.format(se))
    error_list.append(se)
   
    # BACKPROPAGATION

    # Derivative calculation
    chain_1 = np.multiply(np.subtract(y,y_hat),2)
    #print('DERIVATIVE#1: {}'.format(chain_1))

    chain_2 = np.array(sigmoid(np.dot(X,w))*(1-sigmoid(np.dot(X,w))))
    #print('DERIVATIVE#2: {}'.format(chain_2))
  
    chain_3 = X # the derivative of a*x is a!
    #print('DERIVATIVE#3: {}'.format(chain_3))

    # DECLARING NEGATIVE GRADIENT (INCLUDING THE LEARNING RATE)
    neg_grad = -(chain_1*chain_2*chain_3)*learning_rate
    print('GRADIENT: {}'.format(neg_grad))
    #print('NEGATIVE_GRADIENT: {}'.format(neg_grad))


    # UPDATING THE WEIGHTS 
    w = np.subtract(w,neg_grad)
    #print('WEIGHTS: {}'.format(w))
    print('-------')   

plt.plot(epochs_x, error_list)
plt.title('Loss fucntion progression')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()