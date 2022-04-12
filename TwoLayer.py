#%%import
import numpy as np
import matplotlib.pyplot as plt

#%%data load

def convert(imgf, labelf, outf, n):
    f = open(imgf, "rb")
    o = open(outf, "w")
    l = open(labelf, "rb")

    f.read(16)
    l.read(8)
    images = []

    for i in range(n):
        image = [ord(l.read(1))]
        for j in range(28 * 28):
            image.append(ord(f.read(1)))
        images.append(image)

    for image in images:
        o.write(",".join(str(pix) for pix in image) + "\n")
    f.close()
    o.close()
    l.close()


convert("train-images.idx3-ubyte", "train-labels.idx1-ubyte",
        "mnist_train.csv", 60000)
convert("t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte",
        "mnist_test.csv", 10000)

print("Convert Finished!")

#%%
def split_data():
    """
    Load the CIFAR-10 dataset from disk and perform preprocessing to prepare
    it for the two-layer neural net classifier. These are the same steps as
    we used for the SVM, but condensed to a single function.
    """
    train_p = r'D:\卿阁\金毛\机器学习\神经网络\hw1\mnist_train.csv'
    test_p = r'D:\卿阁\金毛\机器学习\神经网络\hw1\mnist_test.csv'

    with open(train_p, encoding='utf-8') as f:
        train_dta = np.loadtxt(f, str, delimiter=",")
        # 取前5行
        print(train_dta[:5])
    train_X=train_dta[:,1:].astype(float)
    train_y=train_dta[:,0].astype(int)

    with open(test_p, encoding='utf-8') as f:
        test_dta = np.loadtxt(f, str, delimiter=",")
        # 取前5行
        print(test_dta[:5])
    test_X = test_dta[:, 1:].astype(float)
    test_y = test_dta[:, 0].astype(int)


    num_validation=test_dta.shape[0]
    num_test=test_dta.shape[0]
    num_training =train_dta.shape[0]-num_validation

    # Subsample the data
    mask = list(range(num_training, num_training + num_validation))
    X_val = train_X[mask]
    y_val = train_y[mask]

    mask = list(range(num_training))
    X_train = train_X[mask]
    y_train = train_y[mask]

    mask = list(range(num_test))
    X_test = test_X[mask]
    y_test = test_y[mask]

    print('Train data shape: ', X_train.shape)
    print('Train labels shape: ', y_train.shape)
    print('Validation data shape: ', X_val.shape)
    print('Validation labels shape: ', y_val.shape)
    print('Test data shape: ', X_test.shape)
    print('Test labels shape: ', y_test.shape)

    # Normalize the data: subtract the mean image
    mean_image = np.mean(X_train, axis=0)
    X_train -= mean_image
    X_val -= mean_image
    X_test -= mean_image

    # # Reshape data to rows
    # X_train = X_train.reshape(num_training, -1)
    # X_val = X_val.reshape(num_validation, -1)
    # X_test = X_test.reshape(num_test, -1)

    return X_train, y_train, X_val, y_val, X_test, y_test

# Invoke the above function to get our data.
X_train, y_train, X_val, y_val, X_test, y_test = split_data()


#%%
def rel_error(x, y):
    """ returns relative error """
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

#%%定义类

class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network. The net has an input dimension of
    N, a hidden layer dimension of H, and performs classification over C classes.
    We train the network with a softmax loss function and L2 regularization on the
    weight matrices. The network uses a ReLU nonlinearity after the first fully
    connected layer.

    In other words, the network has the following architecture:

    input - fully connected layer - ReLU - fully connected layer - softmax

    The outputs of the second fully-connected layer are the scores for each class.
    """

    def __init__(self, input_size, hidden_size, output_size, std=1e-4):
        """
        Initialize the model. Weights are initialized to small random values and
        biases are initialized to zero. Weights and biases are stored in the
        variable self.params, which is a dictionary with the following keys:

        W1: First layer weights; has shape (D, H)
        b1: First layer biases; has shape (H,)
        W2: Second layer weights; has shape (H, C)
        b2: Second layer biases; has shape (C,)

        Inputs:
        - input_size: The dimension D of the input data.
        - hidden_size: The number of neurons H in the hidden layer.
        - output_size: The number of classes C.
        """
        self.params = {}
        self.params['W1'] = std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def loss(self, X, y=None, reg=0.0):
        """
        Compute the loss and gradients for a two layer fully connected neural
        network.

        Inputs:
        - X: Input data of shape (N, D). Each X[i] is a training sample.
        - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
          an integer in the range 0 <= y[i] < C. This parameter is optional; if it
          is not passed then we only return scores, and if it is passed then we
          instead return the loss and gradients.
        - reg: Regularization strength.

        Returns:
        If y is None, return a matrix scores of shape (N, C) where scores[i, c] is
        the score for class c on input X[i].

        If y is not None, instead return a tuple of:
        - loss: Loss (data loss and regularization loss) for this batch of training
          samples.
        - grads: Dictionary mapping parameter names to gradients of those parameters
          with respect to the loss function; has the same keys as self.params.
        """
        # Unpack variables from the params dictionary
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        N, D = X.shape

        ################### Compute the forward pass###############
        scores = None

        # FC1 layer.
        fc1_activation = np.dot(X, W1) + b1

        # Relu layer.
        relu_1_activation = fc1_activation
        relu_1_activation[relu_1_activation < 0] = 0

        # FC2 layer.
        fc2_activation = np.dot(relu_1_activation, W2) + b2

        # Output scores.
        scores = fc2_activation

        # If the targets are not given then jump out, we're done
        if y is None:
            return scores

        # Compute the loss
        loss = 0.0

        # score是N*C矩阵 每个样本对应C个类别的向量

        # Stability fix for softmax scores.
        shift_scores = scores - np.max(scores, axis=1)[:, np.newaxis]############?

        # Calculate softmax scores.
        softmax_scores = np.exp(shift_scores)/np.sum(np.exp(shift_scores), axis=1)[:, np.newaxis]

        # Calculate our cross entropy Loss.
        # y是label 是index 这里选出每个样本真实类别的预测得分 是N维向量
        correct_class_scores = np.choose(y, shift_scores.T)  # Size N vector
        loss = -correct_class_scores + np.log(np.sum(np.exp(shift_scores), axis=1))
        loss = np.sum(loss)

        # Average the loss & add the regularisation loss: lambda*sum(weights.^2).
        loss /= N
        loss += reg * (np.sum(W1 * W1) + np.sum(W2 * W2))#L2 regularity

        # Backward pass: compute gradients
        grads = {}

        # Calculate dSoft - the gradient wrt softmax scores.
        dSoft = softmax_scores
        #将softmax得分矩阵中每个样本的正确标签处-1
        dSoft[range(N), y] = dSoft[range(N), y] - 1#两步求导二合一
        dSoft /= N  # Average over batch.

        # Backprop dSoft to calculate dW2 and add regularisation derivative.
        dW2 = np.dot(relu_1_activation.T, dSoft)
        dW2 += 2 * reg * W2
        grads['W2'] = dW2

        # Backprop dSoft to calculate db2.
        db2 = dSoft * 1
        grads['b2'] = np.sum(db2, axis=0)

        # Calculate dx2 and backprop to calculate dRelu1.
        dx2 = np.dot(dSoft, W2.T)
        relu_mask = (relu_1_activation > 0)
        dRelu1 = relu_mask * dx2

        # Backprop dRelu1 to calculate dW1 and add regularisation derivative.
        dW1 = np.dot(X.T, dRelu1)
        dW1 += 2 * reg * W1
        grads['W1'] = dW1

        # Backprop dRelu1 to calculate db1.
        db1 = dRelu1 * 1
        grads['b1'] = np.sum(db1, axis=0)

        return loss, grads

    def train(self, X, y, X_val, y_val, #mini_batch_size=20,
              learning_rate=1e-3, learning_rate_decay=0.95,
              reg=5e-6, num_iters=100,
              batch_size=200, verbose=False):
        """
        Train this neural network using stochastic gradient descent.

        Inputs:
        - X: A numpy array of shape (N, D) giving training data.
        - y: A numpy array f shape (N,) giving training labels; y[i] = c means that
          X[i] has label c, where 0 <= c < C.
        - X_val: A numpy array of shape (N_val, D) giving validation data.
        - y_val: A numpy array of shape (N_val,) giving validation labels.
        - learning_rate: Scalar giving learning rate for optimization.
        - learning_rate_decay: Scalar giving factor used to decay the learning rate
          after each epoch.
        - reg: Scalar giving regularization strength.
        - num_iters: Number of steps to take when optimizing.
        - batch_size: Number of training examples to use per step.
        - verbose: boolean; if true print progress during optimization.
        """
        num_train = X.shape[0]
        iterations_per_epoch = max(num_train / batch_size, 1)

        # Use SGD to optimize the parameters in self.model
        train_loss_history = []
        test_loss_history=[]
        train_acc_history = []
        val_acc_history = []

        for it in range(num_iters):
            X_batch = None
            y_batch = None

            # Only take the first 'batch_size' elements.
            random_batch = np.random.permutation(num_train)[0:batch_size]
            X_batch = X[random_batch, ...]
            y_batch = y[random_batch]

            # Compute loss and gradients using the current minibatch.
            train_loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
            test_loss,_ = self.loss(X_val, y=y_val, reg=reg)
            train_loss_history.append(train_loss)
            test_loss_history.append(test_loss)

            # Vanilla gradient descent update.
            self.params['W1'] += -grads['W1'] * learning_rate
            self.params['b1'] += -grads['b1'] * learning_rate
            self.params['W2'] += -grads['W2'] * learning_rate
            self.params['b2'] += -grads['b2'] * learning_rate

            if verbose and it % 100 == 0:
                print('iteration %d / %d: loss %f' % (it, num_iters, train_loss))

            # Every epoch, check train and val accuracy and decay learning rate.
            if it % iterations_per_epoch == 0:
                # Check accuracy
                train_acc = (self.predict(X_batch) == y_batch).mean()
                val_acc = (self.predict(X_val) == y_val).mean()
                train_acc_history.append(train_acc)
                val_acc_history.append(val_acc)

                # Decay learning rate
                learning_rate *= learning_rate_decay

        return {
            'train_loss_history': train_loss_history,
            'test_loss_history': test_loss_history,
            'train_acc_history': train_acc_history,
            'val_acc_history': val_acc_history,
        }

    def predict(self, X):
        """
        Use the trained weights of this two-layer network to predict labels for
        data points. For each data point we predict scores for each of the C
        classes, and assign each data point to the class with the highest score.

        Inputs:
        - X: A numpy array of shape (N, D) giving N D-dimensional data points to
          classify.

        Returns:
        - y_pred: A numpy array of shape (N,) giving predicted labels for each of
          the elements of X. For all i, y_pred[i] = c means that X[i] is predicted
          to have class c, where 0 <= c < C.
        """
        y_pred = None

        # Get the index of highest score, this is our predicted class.
        y_pred = np.argmax(self.loss(X), axis=1)

        return y_pred

#%%开始真正的数据!
input_size = 784
hidden_size = 500
num_classes = 10
net = TwoLayerNet(input_size, hidden_size, num_classes)

# Train the network
stats = net.train(X_train, y_train, X_val, y_val,
            num_iters=1000, batch_size=200,
            learning_rate=0.01, learning_rate_decay=0.95,
            reg=0.01, verbose=True)

# Predict on the validation set
val_acc = (net.predict(X_val) == y_val).mean()
print('Validation accuracy: ', val_acc)
#%%grid search
input_size = 784
num_classes = 10

for learning_rate in [0.001,0.01,0.05]:
    for reg in [0.005,0.01,0.05]:
        for hidden_size in [50,100,500,1000]:
            net = TwoLayerNet(input_size, hidden_size, num_classes)
            # Train the network
            stats = net.train(X_train, y_train, X_val, y_val,
                              num_iters=1000, batch_size=200,
                              learning_rate=learning_rate, learning_rate_decay=0.95,
                              reg=reg, verbose=True)
            # Predict on the validation set
            val_acc = (net.predict(X_val) == y_val).mean()
            print('parameters',learning_rate,reg,hidden_size,'Validation accuracy: ', val_acc)


#%%curve
# Plot the loss function and train / validation accuracies
plt.subplot(2, 1, 1)
plt.plot(stats['train_loss_history'])
plt.plot(stats['test_loss_history'])
plt.legend(['Training Set',"Validation Set"])
plt.title('Loss history')
plt.xlabel('Iteration')
plt.ylabel('Loss')

plt.subplot(2, 1, 2)
plt.plot(stats['train_acc_history'], label='train')
plt.plot(stats['val_acc_history'], label='val')
plt.title('Classification accuracy history')
plt.legend(['Training Set',"Validation Set"])
plt.xlabel('Epoch')
plt.ylabel('Clasification accuracy')
plt.show()

#%%visualize_grid
from math import sqrt, ceil

def visualize_grid(Xs, ubound=255.0, padding=1):
  """
  Reshape a 3D tensor of image data to a grid for easy visualization.

  Inputs:
  - Xs: Data of shape (N, H, W)
  - ubound: Output grid will have values scaled to the range [0, ubound]
  - padding: The number of blank pixels between elements of the grid
  """
  (N, H, W) = Xs.shape
  grid_size = int(ceil(sqrt(N)))
  grid_height = H * grid_size + padding * (grid_size - 1)
  grid_width = W * grid_size + padding * (grid_size - 1)
  grid = np.zeros((grid_height, grid_width))
  next_idx = 0
  y0, y1 = 0, H
  for y in range(grid_size):
    x0, x1 = 0, W
    for x in range(grid_size):
      if next_idx < N:
        img = Xs[next_idx]
        low, high = np.min(img), np.max(img)
        grid[y0:y1, x0:x1] = ubound * (img - low) / (high - low)
        # grid[y0:y1, x0:x1] = Xs[next_idx]
        next_idx += 1
      x0 += W + padding
      x1 += W + padding
    y0 += H + padding
    y1 += H + padding
  # grid_max = np.max(grid)
  # grid_min = np.min(grid)
  # grid = ubound * (grid - grid_min) / (grid_max - grid_min)
  return grid

# Visualize the weights of the network

def show_net_weights(net):
    W1 = net.params['W1']
    W1 = W1.reshape(28,28, -1).transpose(2,0,1)
    plt.imshow(visualize_grid(W1, padding=3).astype('uint8'))
    plt.gca().axis('off')
    plt.show()

show_net_weights(net)

#%%final test
test_acc = (net.predict(X_test) == y_test).mean()
print('Test accuracy: ', test_acc)

#%%
import pickle

with open('forsave.pkl', 'wb') as file:
    pickle.dump(net.params, file)

#%%
with open('forsave.pkl', 'rb') as file:
    read= pickle.load(file)

print(read)