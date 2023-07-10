import numpy as np
import os
import network
import mnist_loader
from  PIL import Image,ImageOps

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
net = network.Network([784, 30, 10])
net.biases,net.weights = net.SGD(training_data, 50, 10, 3,test_data)
while 1 :
    f = input("图片\n")
    img = Image.open("{0}".format(f))
    img = img.resize((28, 28))
    img = img.convert('L')
    img = ImageOps.invert(img)
    img = img.point(lambda x: x /255)
    img = np.array(img)
    img = net.center_digit(img,28)
    print(img)
    img = img.reshape((784,1))
    net.detect(img)


