import tensorflow as tf
import os
import csv
import numpy as np

_test_data = None

_training_data = None
_training_labels = None
def get_test_data():    
    print("data processing")
    training_path = '%s/data/train.csv'%( os.path.dirname(os.path.abspath(__file__)) )
    test_path = '%s/data/test.csv'%( os.path.dirname(os.path.abspath(__file__)) )
        
    # read data
    train_data = []
    i = 0
    with open(training_path) as training_file:
        lines = csv.reader(training_file)
        isValid = False
        for line in lines:
            if isValid:
                train_data.append(line)
                i += 1
                if i >= 200:
                    break  
            else:
                isValid = True                   
    #data split
    train_data = np.array(train_data).astype(np.int16)#str to float
    global _training_data
    _training_data = ( (train_data[:,1:] > 0) * 1.0 ).astype(np.float32).reshape(-1,28,28,1)#delete the first column,which are correct_labels. 
        
    data_size =  _training_data.shape[0]    
    correct_labels = train_data[:,0].reshape(data_size,1)#get the correct correct_labels
    global _training_labels
    _training_labels = np.zeros([data_size,10],dtype = np.float32)
    for i in xrange(data_size):
        real_class = correct_labels[i][0]
        _training_labels[i,real_class] = 1.0
            
              
        # read data              
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
        
    print("finished data processing")

def get_accuracy(ys,y_pre):
    correct_pre = tf.equal(tf.argmax(ys,axis=1),tf.argmax(y_pre,axis=1))
    accuracy = tf.reduce_mean(tf.cast(correct_pre,tf.float32))
    accuracy = tf.Session().run(accuracy)
    return accuracy


def write_to_file(path,data):
    f = open(path,'w+')
    f.write('ImageId'+','+'Label'+'\n')
    size = data.shape[0]
    for i in xrange(size):
        f.write(str(i+1)+','+str(data[i])+'\n')
    f.close()


get_test_data()

saver = tf.train.import_meta_graph('./model_kaggle_400/kaggle.meta')
graph = tf.get_default_graph()

with tf.Session() as sess:
    saver.restore(sess, './model_kaggle_400/kaggle')
    
    #restore from model
    xs = graph.get_tensor_by_name('input_data:0')
    ys = graph.get_tensor_by_name('input_label:0')
    prediction = graph.get_tensor_by_name('prediction:0')
    drop_out = graph.get_tensor_by_name('drop_out:0')
    
    #using restored model to prediction 
    y_pre = sess.run(prediction,feed_dict={xs:_training_data,drop_out:1.0})
    accuracy = get_accuracy(_training_labels,y_pre)
    print('accuracy in training set: %s' %accuracy)
    
    pre_softmax = sess.run(prediction,feed_dict={xs:_test_data,drop_out:1.0})
    pre_label = tf.argmax(pre_softmax,axis=1)
    
    #save prediction
    write_to_file('test_prediction.csv',sess.run(pre_label))
  
    
