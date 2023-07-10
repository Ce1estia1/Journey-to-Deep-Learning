import numpy as np
import random


class Network(object):
    def __init__(self,sizes):
        self.numlayers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y,1) for y in sizes[1:]]
        self.weights = [np.random.randn(x,y) for y,x in list((sizes[:-1],sizes[1:]))]
    def sigmoid(self,z):
        return 1.0/(1.0+np.exp(-z))

    def center_digit(self,digit, size):
        # 找到数字的边界框
        rows, cols = np.where(digit)
        rmin, rmax = rows.min(), rows.max()
        cmin, cmax = cols.min(), cols.max()
        # 计算数字图像的中心点
        rcenter, ccenter = (rmax + rmin) / 2, (cmax + cmin) / 2
        # 计算输出图像的中心点
        center = np.array([size / 2, size / 2])
        # 计算数字图像在输出图像中的左上角位置
        topleft = np.round(center - [rcenter, ccenter]).astype(int)
        # 创建输出图像
        output = np.zeros((size, size))
        bottomright = topleft + [digit.shape[0], digit.shape[1]]
        # 检查左上角位置是否超出输出图像边界
        if topleft[0] < 0:
            topleft[0] = 0
        if topleft[1] < 0:
            topleft[1] = 0
        # 检查右下角位置是否超出输出图像边界
        if bottomright[0] > size:
            bottomright[0] = size
        if bottomright[1] > size:
            bottomright[1] = size
        # 将数字图像插入到输出图像中心
        output[topleft[0]:bottomright[0], topleft[1]:bottomright[1]] = digit[ 0: (bottomright[0] - topleft[0]),0: (bottomright[1] - topleft[1])]
        return output
    def sigmoid_prime(self,z):
        return self.sigmoid(z)*(1-self.sigmoid(z))

    def evaluate(self,test_data):
        test_results = [(np.argmax(self.feedforward(x)),y )for (x,y) in test_data]
        #print(test_results[0])
        return sum(int (x == y) for (x,y) in test_results)
    def cost_derivative(self,output_acitivations,y):
        return (output_acitivations - y)
    def feedforward(self,a):
        for b, w in list(zip(self.biases,self.weights)):
            a = self.sigmoid(np.dot(w,a) + b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data):
        if test_data: n_test= len(test_data)
        n = len(training_data)
        for j in range(epochs):
            if j >200 and j< 300 :
                eta = 0.1
            if j>300:
                eta = 0.01
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0,n,mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch,eta)
            if test_data:
               print("Epoch {0}: {1} / {2}".format(j,self.evaluate(test_data), n_test))
            else:
               print("Epoch {0} complete".format(j))
        return (self.biases, self.weights)

    def detect(self,detect_data):
        self.evaluates(detect_data)

    def evaluates(self,test_data):
        test_result = np.argmax(self.feedforward(test_data))
        print(test_result)

    def update_mini_batch(self, mini_batch, eta):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x,y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in list(zip(nabla_b, delta_nabla_b))]
            nabla_w = [nw + dnw for nw, dnw in list(zip(nabla_w, delta_nabla_w))]
        self.weights = [w - (eta / len(mini_batch)) * nw for w, nw in list(zip(self.weights, nabla_w))]
        self.biases = [b - (eta / len(mini_batch)) * nb for b, nb in list(zip(self.biases, nabla_b))]

    def backprop(self, x, y ):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        activation = x
        activations = [x]
        zs = []

        for b, w in list(zip(self.biases,self.weights)):
            z = np.dot(w,activation)+b
            zs.append(z)
            activation = self.sigmoid(z)
            activations.append(activation)

        delta = self.cost_derivative(activations[-1],y)*self.sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        for l in range(2,self.numlayers):
            z = zs[-l]
            sp = self.sigmoid_prime(z)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())
        return(nabla_b, nabla_w)