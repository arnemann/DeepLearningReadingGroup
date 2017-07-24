# based on the TensorFlow "Deep MNIST for Experts" tutorial

import tensorflow as tf

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal

def specifyGraph(weight_decay=2e-1):
    
    graph = tf.Graph()
    
    with graph.as_default():
        
        x = tf.placeholder(tf.float32, shape=[None, 784],name="inputs")
        y_ = tf.placeholder(tf.float32, shape=[None, 10],name="targets")

        x_image = tf.reshape(x, [-1, 28, 28, 1],name="reshaped_input")

        keep_prob = tf.placeholder(tf.float32,name="dropout_keep_probability")

        ### Convolution 1
        W_conv1 = weight_variable([9, 9, 1, 32],'weights_convolution_1')
        b_conv1 = bias_variable([32],'biases_convolution_1')

        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1,'output_convolution_1') 
                                     + b_conv1,
                            name="activations_convolution_1")

        h_pool1 = max_pool_2x2(h_conv1,"activations_pool_1")

        ### Convolution 2

        W_conv2 = weight_variable([5, 5, 32, 64],'weights_convolution_2')
        b_conv2 = bias_variable([64], 'biases_convolution_2')

        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2,'output_convolution_2') 
                                     + b_conv2,
                            name="activations_convolution_2")

        h_pool2 = max_pool_2x2(h_conv2,"activations_pool_2")

        ### Fully-Connected 1

        W_fc1 = weight_variable([7 * 7 * 64, 1024],'weights_fullyconnected_1')
        b_fc1 = bias_variable([1024],'biases_fullyconnected_1')

        h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64],
                                 name="reshapedactivations_hidden_2")
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1,
                                    name="output_fullyconnected_1") 
                           + b_fc1,
                          name="activations_fullyconnected_1")


        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob,
                                  name="droppedactivations_fullyconnected_1")

        ### Fully-Connected 2

        W_fc2 = weight_variable([1024, 10],"weights_fullyconnected_2")
        b_fc2 = bias_variable([10], "biases_fullyconnected_2")

        y_conv = tf.add(tf.matmul(h_fc1_drop, W_fc2,
                          name="outputs_fullyconnected_2"),
                        b_fc2,name="activations_fullyconnected_2")

        cross_entropy = tf.reduce_mean(
                            tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv,
                                                                       name="crossentropy_cost"),
                                name="mean_crossentropy_cost")

        weight_decay = tf.reduce_sum([tf.nn.l2_loss(weight_decay*weight) # .2, 10k; .1, 25k
                              for weight in [W_conv1,W_conv2,W_fc1,W_fc2]])

        total_cost = cross_entropy + weight_decay

        train_step = tf.train.AdamOptimizer(1e-4).minimize(total_cost)

        correct_prediction = tf.equal(tf.argmax(y_conv, 1,
                                               name="MAPestimate_index"), 
                                      tf.argmax(y_, 1,
                                               name="label_index"),
                                     name="MAP_iscorrect")

        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32),
                             name="fraction_correct")

        operations = {'inputs':x,
                      'targets':y_,
                      'keep_prob':keep_prob,
                      'accuracy':accuracy,
                      'output':y_conv,
                      'train_step':train_step,
                      'filters':W_conv1
                      }
    
    return graph, operations

def weight_variable(shape,name):
    initial = tf.truncated_normal(shape, stddev=0.1, name=name)
    return tf.Variable(initial)

def bias_variable(shape,name):
    initial = tf.constant(0.1, shape=shape, name=name)
    return tf.Variable(initial)

def conv2d(x, W,name):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME', name=name)

def max_pool_2x2(x,name):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME', name=name)

def plotFilters(filterBank,title=""):
    numFilters = filterBank.shape[-1]
    plt.figure(figsize=(6,6))

    for idx in range(numFilters):
        ax = plt.subplot(5,7,idx+1)
        filtr = filterBank[:,:,0,idx]
        ax.imshow(filtr,cmap='Greys',interpolation='None')
        ax.set_xticks([]); ax.set_yticks([]);
    plt.suptitle(title+" Filters",
                fontsize='xx-large',fontweight='bold');
    
def plotAutoCorrs(weights,names):
    
    plt.figure(figsize=(16,4));
    plt.subplot(1,len(weights),1)
    
    for idx,(weight,name) in enumerate(zip(weights,names)):
        ax = plt.subplot(1,len(weights),idx+1)
        plotAutoCorr(ax,weight,name)
    
    plt.tight_layout()
    plt.suptitle("Weight Auto-Correlations",
                fontsize='xx-large',fontweight='bold',
                 y=1.05,
                )
    return

def plotAutoCorr(ax,weight,name):
    
    auto_corr = scipy.signal.correlate2d(weight,weight,mode='same')
    auto_corr_shape = auto_corr.shape
    auto_corr = auto_corr.ravel()
    auto_corr[np.argmax(auto_corr)] = 0
    auto_corr = auto_corr.reshape(auto_corr_shape)
    
    ax.imshow(np.triu(auto_corr));
    ax.set_xticks([]); ax.set_yticks([]);
    plt.title(name,
             fontsize='x-large',
             fontweight='bold')
    return