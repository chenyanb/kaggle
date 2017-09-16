# View more python tutorial on my Youtube and Youku channel!!!

# Youtube video tutorial: https://www.youtube.com/channel/UCdyjiB5H8Pu7aDTNVXTTpcg
# Youku video tutorial: http://i.youku.com/pythontutorial

"""
Please note, this code is only for python 3+. If you are using python 2+, please modify the code accordingly.
"""
from __future__ import print_function
import tensorflow as tf
import numpy as np
import os
import csv
# from tensorflow.examples.tutorials.mnist import input_data
# number 1 to 10 data
# mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

_training_data = None
_training_labels = None
 
_data_size = None
 
_dev_data = None
_dev_labels = None
 
_test_data = None

_learning_rate = 1e-3

def compute_accuracy():
#     global prediction
#     global _dev_data,_dev_labels
    y_pre = sess.run(prediction, feed_dict={xs: _dev_data, keep_prob: 1})
    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(_dev_labels,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32),name='accuracy')
    result = sess.run(accuracy, feed_dict={xs: _dev_data, ys: _dev_labels, keep_prob: 1})
    return result

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    # stride [1, x_movement, y_movement, 1]
    # Must have strides[0] = strides[3] = 1
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    # stride [1, x_movement, y_movement, 1]
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

def data_processing():
        print("data processing")
        training_path = '%s/data/train.csv'%( os.path.dirname(os.path.abspath(__file__)) )
        test_path = '%s/data/test.csv'%( os.path.dirname(os.path.abspath(__file__)) )
        
        # read data
        train_data = []
        with open(training_path) as training_file:
            lines = csv.reader(training_file)
            isValid = False
            for line in lines:
                if isValid:
                    train_data.append(line)  
                else:
                    isValid = True                   
        #data split
        train_data = np.array(train_data).astype(np.int16)#str to float
        global _training_data
        _training_data = ( (train_data[:,1:] > 0) * 1.0 ).astype(np.float32).reshape(-1,28*28*1)#delete the first column,which are correct_labels. 
        
        data_size =  _training_data.shape[0]    
        correct_labels = train_data[:,0].reshape(data_size,1)#get the correct correct_labels
        global _training_labels
        _training_labels = np.zeros([data_size,10],dtype = np.float32)
        for i in xrange(data_size):
            real_class = correct_labels[i][0]
            _training_labels[i,real_class] = 1.0
                  
        test_data = []
        with open(test_path) as test_file:
            lines = csv.reader(test_file)
            isValid = False
            for line in lines:
                if isValid:
                    test_data.append(line)
                else:
                    isValid = True
        
        #data split
        test_data = np.array(test_data).astype(np.int16)
        global _test_data
        _test_data = ( (test_data > 0) * 1.0 ).reshape([-1,28,28,1])#delete the first column,which are correct_labels. 
        
        random_dev_index = np.random.randint(low = 0,high = data_size,size = 100) 
        global _dev_data        
        _dev_data = _training_data[random_dev_index].reshape(100,28,28,1)
        global _dev_labels
        _dev_labels = _training_labels[random_dev_index].reshape(100,10)
        
        global _training_data
        _training_data = np.delete(_training_data, random_dev_index,axis=0).reshape(-1,28,28,1)
        global _training_labels
        _training_labels = np.delete(_training_labels,random_dev_index,axis=0).reshape(-1,10)
        
        global _data_size
        _data_size = _training_data.shape[0]
        print("finished data processing")

def get_data_batchs(data=_training_data, batch_size=100):
    size = data.shape[0]
    for i in range(size//100):
        if (i+1)*batch_size > size:
            yield data[i*batch_size:].reshape(-1,28,28,1)
        else:
            yield data[i*batch_size:(i+1)*batch_size].reshape(-1,28,28,1)

def get_labels_batchs(data=_training_labels, batch_size=100):
    size = data.shape[0]
    for i in range(size//100):
        if (i+1)*batch_size > size:
            yield data[i*batch_size:].reshape(-1,10)
        else:
            yield data[i*batch_size:(i+1)*batch_size].reshape(-1,10)

#default 1e-5
def writeToFile(data):
    f = open('test_kaggle_400.csv', 'w+')
    f.write('ImageId'+','+'Label'+'\n')
    size = data.shape[0]
    for i in xrange(size):
        f.write(str(i+1)+','+str(data[i])+'\n')
    f.close()
# define placeholder for inputs to network
data_processing()
xs = tf.placeholder(tf.float32, [None, 28,28,1],name='input_data')   # 28x28
ys = tf.placeholder(tf.float32, [None, 10],name="input_label")
keep_prob = tf.placeholder(tf.float32,name='drop_out')
# x_image = tf.reshape(xs, [-1, 28, 28, 1])
# print(x_image.shape)  # [n_samples, 28,28,1]

## conv1 layer ##
W_conv1 = weight_variable([3,3, 1,32]) # patch 3x3, in size 1, out size 32
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(xs, W_conv1) + b_conv1) # output size 28x28x32
h_pool1 = max_pool_2x2(h_conv1)#14 * 14 * 32                                         # output size 14x14x32

## conv2 layer ##
W_conv2 = weight_variable([3,3, 32, 64]) # patch 5x5, in size 32, out size 64
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2) # output size 14x14x64
h_pool2 = max_pool_2x2(h_conv2)                                         # output size 7x7x64

## fc1 layer ##
W_fc1 = weight_variable([7*7*64, 1024])
b_fc1 = bias_variable([1024])
# [n_samples, 7, 7, 64] ->> [n_samples, 7*7*64]
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

## fc2 layer ##
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
# h_fc2 = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
h_fc2 = tf.add(tf.matmul(h_fc1_drop, W_fc2),b_fc2,name='fc_out')


prediction = tf.nn.softmax(h_fc2,name='prediction')


# the error between prediction and real data
# cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),
#                                               reduction_indices=[1]))       # loss

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = h_fc2,
                                                             labels = ys),name='cross_entropy')

train_step = tf.train.AdamOptimizer(_learning_rate).minimize(cross_entropy,name='train_step')

sess = tf.Session()
# important step
# tf.initialize_all_variables() no long valid from
# 2017-03-02 if using tensorflow >= 0.12
init = tf.global_variables_initializer()
sess.run(init)
saver = tf.train.Saver()
count = 0
global _learning_rate
for i in range(150000):
    global _data_size
    batch_index = np.random.randint(low=0,high= _data_size,size = 400)#################################     
    batch_xs = _training_data[batch_index].reshape(-1,28,28,1)
    batch_ys = _training_labels[batch_index].reshape(-1,10)

    sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: 0.5})
    if i % 50 == 0:
        accuracy = compute_accuracy()
        print(accuracy)
        if accuracy >= 0.99:
            count += 1
        else:
            count = 0
        
        if accuracy >= 0.99:
            _lerning_rate = 1e-6
        elif accuracy >= 0.98:
            _learning_rate = 1e-5
        elif accuracy >= 0.97:
            _learning_rate =  1e-4
                  
        if count >= 10:
            print('num of iteration:%s' %(i))
            break
        
        
global _test_data
predict = sess.run(prediction,feed_dict={xs:_test_data,keep_prob: 1.0})
res = np.argmax(predict,axis=1)
writeToFile(res)

save_path = saver.save(sess,save_path='./model_kaggle_400/kaggle')
print('save model to : ./model_kaggle_400/kaggle')
print("batch=400")





