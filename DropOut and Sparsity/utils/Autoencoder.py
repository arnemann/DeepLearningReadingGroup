import tensorflow as tf

import os

import numpy as np

import matplotlib

matplotlib.use('agg')

import matplotlib.pyplot as plt


def train(graph, initializer, input_variable, weights, biases, output_op, keep_prob, optimizer, \
                     cost, saver, mnist, training_parameters):

    batch_size = training_parameters["batch_size"]
    num_epochs = training_parameters["num_epochs"]
    display_every = training_parameters["display_every"]
    output_folder = training_parameters["output_folder"]
    show_final = training_parameters["show_final"]
    keep_prob_value = 1-training_parameters["dropout_rate"]

    with tf.Session(graph=graph) as sess:

        sess.run(initializer)
        batches_per_epoch = int(mnist.train.num_examples/batch_size)

        # train
        for epoch in range(num_epochs):
            for i in range(batches_per_epoch):
                test_batch, _ = mnist.train.next_batch(batch_size)
                sess.run([optimizer], feed_dict={input_variable: test_batch,
                                                keep_prob: keep_prob_value}
                                                )

            if (display_every is not None) and (epoch % display_every == 0):
                test_c, = sess.run([cost], feed_dict={input_variable: test_batch})
                valid_batch, _ =  mnist.validation.next_batch(batch_size)
                valid_c, = sess.run([cost], feed_dict={input_variable: valid_batch})
                print("Epoch: ", str(epoch+1).zfill(4), "\t",
                      "test cost=", "{:.9f}".format(test_c), "\t",
                      "val cost=",  "{:.9f}".format(valid_c), "\t")
        if output_folder is not None:
            modelFile = saver.save(sess, output_folder+'model')
        else:
            modelFile = saver.save(sess, 'models/lastRun/model')

        modelFile = os.path.split(modelFile)[0]

        if show_final:
            examples_to_show = 10
            encode_decode = sess.run(output_op,
                                 feed_dict={input_variable: mnist.validation.images[:examples_to_show]})
            f, a = plt.subplots(2, examples_to_show, figsize=(6, 2))
            plt.suptitle("test set")
            for i in range(examples_to_show):
                a[0][i].imshow(np.reshape(mnist.validation.images[i], (28, 28)))
                a[1][i].imshow(np.reshape(encode_decode[i], (28, 28)))
                for row in range(2):
                    a[row][i].get_xaxis().set_visible(False)
                    a[row][i].get_yaxis().set_visible(False)

            plt.tight_layout()

        encoder_weights = weights['encoder_weights_00'].eval(session=sess)
        encoder_biases = biases['encoder_biases_00'].eval(session=sess)

    return modelFile, encoder_weights,encoder_biases

def getCosts(graph,initializer,saver,input_variable,mnist,modelFolder):

    with tf.Session(graph=graph) as sess:

        saver.restore(sess,
                        tf.train.latest_checkpoint(modelFolder))
        mse_cost = graph.get_collection('costs')[0]
        train_batch, _ = mnist.train.next_batch(10000)
        valid_batch, _ = mnist.validation.next_batch(10000)

        train_cost = sess.run([mse_cost], feed_dict={input_variable:train_batch})
        valid_cost = sess.run([mse_cost], feed_dict={input_variable:valid_batch})

    return train_cost, valid_cost

def setup(hyperparameters):
    """create a symmetric autoencoder with homogeneous activation functions,
       with hyperparameter dictionary as setup by createHyperparameterDict"""

    graph = tf.Graph()
    with tf.device("/gpu:0"):
        with graph.as_default():
            input, weights, biases, encoder, decoder,\
                    all_layers, keep_prob, target, cost = specifyNetwork(hyperparameters)

            optimizer = specifyOptimizer(cost,hyperparameters)

            initializer = tf.global_variables_initializer()

            saver = setupSaver(weights,biases)

    return graph, initializer, input, weights, biases, encoder, decoder,\
                all_layers, keep_prob, optimizer, cost, saver

def specifyNetwork(hyperparameters):

    weight_initializer = hyperparameters['weight_initializer']
    widths = hyperparameters['widths']
    nonlinearity = hyperparameters['nonlinearity']
    weight_decay_rate = hyperparameters['weight_decay_rate']
    shared_bias = hyperparameters['shared_bias']

    mean_image = np.loadtxt('/home/charlesfrye/Notebooks/DropOut/mean_image.txt')

    weights = {}
    biases = {}

    if weight_initializer is None:
        weight_initializer = initializeWeights

    for layer,_ in enumerate(widths[:-1]):
        width_in = widths[layer]
        width_out = widths[layer+1]

        weights['encoder_weights_'+str(layer).zfill(2)], \
            biases['encoder_biases_'+str(layer).zfill(2)] = weight_initializer(width_in, width_out,shared_bias)

    for layer,_ in enumerate(widths[1:]):
        width_in = widths[-1-layer]
        width_out = widths[-1-layer-1]

        weights['decoder_weights_'+str(layer).zfill(2)], \
            biases['decoder_biases_'+str(layer).zfill(2)] = weight_initializer(width_in, width_out,shared_bias)

    input = tf.placeholder("float", [None, widths[0]],name="inputs")

    zero_centered_input = tf.subtract(input,
                            tf.constant(mean_image,name="mean_img",dtype=tf.float32),
                                            "mean_subtracted_image")

    keep_prob = tf.placeholder_with_default(1.0, shape=(),
                                            name='hidden_keep_rate')

    encoder_states = makeNet(zero_centered_input, weights, biases, nonlinearity, 'encoder', keep_prob, hyperparameters)
    decoder_states = makeNet(encoder_states[-1], weights, biases, nonlinearity, 'decoder', keep_prob, hyperparameters)

    network_states = encoder_states+decoder_states

    encode_op = encoder_states[-1]
    decode_op = decoder_states[-1]

    prediction = decode_op
    target = input

    mse_cost = tf.reduce_mean(tf.pow(target-prediction,2,name="squared_error"),
                                name="mean_squared_error_cost")

    tf.add_to_collection("costs",mse_cost)

    if weight_decay_rate is not None:
        weight_decay_rate = tf.constant(weight_decay_rate, name="weight_decay_rate")
        
        for weight_key,weight_matrix in weights.items():
            layer_size = float(weight_matrix.shape[1].value)
            layerwise_weight_decay = tf.divide(weight_decay_rate,layer_size,
                                               name=weight_key+"_weight_decay")

            layerwise_weight_cost = tf.multiply(layerwise_weight_decay,
                                                tf.nn.l2_loss(weight_matrix,
                                               name = weight_key+"_l2_loss"),
                                                 name="normalized_"+weight_key+"_l2_loss")
            tf.add_to_collection("costs",layerwise_weight_cost)

    total_cost = tf.reduce_sum(tf.get_collection("costs"),name="total_cost")

    return input, weights, biases, encode_op, decode_op, network_states, keep_prob, target, total_cost

def makeNet(input, weights, biases, nonlinearity, tag, keep_prob , hyperparameters):

    weight_keys = sorted([key for key in weights.keys() if tag in key])
    bias_keys = sorted([key for key in biases.keys() if tag in key])

    network_states = [input]

    for layer,_ in enumerate(weight_keys):
        ws = weights[weight_keys[layer]]
        bs = biases[bias_keys[layer]]

        next_state = nonlinearity(tf.add(tf.matmul(network_states[-1],ws),
                            bs,name=tag+"_prenonlinearity_"+str(layer+1).zfill(2)),
                     name=tag+"_activations_"+str(layer+1).zfill(2))

        if (layer == len(weight_keys)-1) & (tag == 'decoder') : # during output step
            pass #no dropout
        else:
            next_state = tf.nn.dropout(next_state, keep_prob = keep_prob,
                                           name=tag+"_dropped_"+str(layer+1).zfill(2))

        network_states.append(next_state)

    return network_states

def specifyOptimizer(cost,hyperparameters):

    optimizer = hyperparameters['optimizer']
    learning_rate_decay = hyperparameters['learning_rate_decay']
    learning_rate = hyperparameters['learning_rate']
    momentum = hyperparameters['momentum']
    use_nesterov=hyperparameters['use_nesterov']
    batch_size = hyperparameters['batch_size']

    batch = tf.Variable(0)

    if learning_rate_decay is not None:
        learning_rate = tf.train.exponential_decay(
          learning_rate,            # Base learning rate.
          batch * batch_size,       # Current index into the dataset.
          len(mnist.train.images),  # Decay step.
          learning_rate_decay,      # Decay rate.
          staircase=True)

    if optimizer is tf.train.MomentumOptimizer:
        optimizer = optimizer(learning_rate,
                              momentum,use_nesterov=use_nesterov)
    else:
        optimizer = optimizer(learning_rate)

    optimizer = optimizer.minimize(cost, global_step = batch)

    return optimizer

def setupSaver(weights_dict, biases_dict):

    listedWeights = list(weights_dict.values())
    listedBiases = list(biases_dict.values())

    saver = tf.train.Saver(listedWeights+listedBiases)

    return saver

#def initializeWeights(width_in, width_out):
#
#    weightMatrix = tf.Variable(tf.random_normal([width_in, width_out]),name="weights")
#    biasVector = tf.Variable(tf.zeros([width_out]),name="biases")
#
#    return weightMatrix, biasVector

def initializeWeights_ReLU(width_in, width_out, shared_bias):
    weightMatrix = tf.Variable(tf.truncated_normal([width_in, width_out],stddev=0.1),name="weights")
    biasVector = tf.Variable(tf.multiply(tf.constant(shared_bias,name="shared_bias_value"),
                                         tf.ones([width_out],name="unit_bias"),name="biases"))
    return weightMatrix, biasVector

def initializeWeights_shared_bias(width_in,width_out,shared_bias):
    weightMatrix = tf.Variable(tf.truncated_normal([width_in, width_out],stddev=0.1),name="weights")
    biasVector = tf.Variable(tf.multiply(tf.constant(shared_bias,name="shared_bias_value"),
                                         tf.ones([width_out],name="unit_bias"),name="biases"))

    return weightMatrix, biasVector

def createHyperparameterDict(widths=[784,256], weight_initializer=None,
                             nonlinearity=tf.nn.sigmoid,
                             learning_rate=0.001, weight_decay_rate=None, learning_rate_decay=None,
                             optimizer=tf.train.AdamOptimizer, use_nesterov=False, momentum=0.9,
                             batch_size=256,
                             shared_bias=-0.1,
                             num_epochs=11,
                             dropout_rate=0.0):

    return locals()

def createHP_Dict(dropout_rate,weight_decay_rate):

    baseDict = {'batch_size': 32,
              'learning_rate': 1.0,
              'learning_rate_decay': None,
              'momentum': 0.9,
              'nonlinearity': tf.nn.relu,
              'optimizer': tf.train.MomentumOptimizer,
              'shared_bias': -0.1,
              'use_nesterov': True,
              'weight_initializer': initializeWeights_shared_bias,
              'widths': [784, 256]}

    baseDict['dropout_rate'] = dropout_rate
    baseDict['weight_decay_rate'] = weight_decay_rate

    return createHyperparameterDict(**baseDict)

def getActivations(graph,input_variable,encode_op,saver,file,mnist,numExamples=5000):
    """returns the hidden unit activations on validation set of size numExamples
        as a numExamples by numUnits array"""
    with tf.Session(graph=graph) as sess:

        saver.restore(sess,tf.train.latest_checkpoint(file))

        valid_batch, _ =  mnist.validation.next_batch(numExamples)

        activations, = sess.run([encode_op], feed_dict= {input_variable: valid_batch})

    return activations

def getReconstructions(graph,input_variable,output_op,saver,file,mnist,numExamples=100):
    """ returns numExamples reconstructions from validation set by given model"""
    with tf.Session(graph=graph) as sess:

        saver.restore(sess,
                      tf.train.latest_checkpoint(file))

        batch, _ =  mnist.validation.next_batch(numExamples)

        recons, = sess.run([output_op], feed_dict= {input_variable: batch})

    return recons

def retrieveWeights(graph,saver,file):
    """ returns the weights for given model as a list of 2-d arrays"""
    with tf.Session(graph=graph) as sess:

        saver.restore(sess,
                      tf.train.latest_checkpoint(file))

        final_weights, = [variable for variable in tf.global_variables() if 'weights:0' in variable.name]
        final_weights = final_weights.eval()

    inputSize, numUnits = final_weights.shape
    inputSideLength = int(np.sqrt(inputSize))

    reshaped_weights = [np.reshape(final_weights[:,idx],
                                   (inputSideLength,inputSideLength))
                                for idx in range(numUnits)]

    return reshaped_weights
