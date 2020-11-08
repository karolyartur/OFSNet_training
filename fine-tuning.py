import tensorflow as tf
import numpy as np
import os
import sys
import math
import tensorflow.contrib.slim as slim
import cv2 as cv
import matplotlib.pyplot as plt

# Import the modified definition of the Inception model from the obstacle-detection repository
sys.path.append(os.path.join(os.path.dirname(os.path.abspath('__file__')), '..', 'obstacle-detection'))
import models.inception

# Data augmentation
def augment(image, mask):
    '''Data augmentation
    
    Function for the augmentation of the training data (random contrast and saturation change and random vertical flip)
    '''
    image = tf.image.random_contrast(image, 0.8, 1.2)
    image = tf.image.random_saturation(image, 0.8, 1.2)
    mask = tf.reshape(mask,[mask.shape[0],mask.shape[1],1])
    conc = tf.image.random_flip_left_right(tf.concat([image,mask], 2))
    image, mask = tf.split(conc, [3,1], axis=2)
    return image, mask

# Parameters
batch_size = 20  # Number of inputs in a batch
iterations = 2000  # Number of training steps
saves = 500  # Number of iterations after wich the model is saved and validation is performed
training_data_dir = os.path.join(os.path.dirname(os.path.abspath('__file__')), 'data', 'train')  # Directory containing training data
valid_data_dir = os.path.join(os.path.dirname(os.path.abspath('__file__')), 'data', 'valid')  # Directory containing validation data

# Function for yielding training data
def get_data(d):
    data_dirs = os.listdir(d)
    directory = np.random.choice(data_dirs)
    inputs = np.load(d+'/%s/np_arrays/inputs.npy' % directory)
    labels = np.load(d+'/%s/np_arrays/masks.npy' % directory)
    for i in range(len(inputs)):
        yield (inputs[i], labels[i+1])

# Function for yielding validation data
def test_data(d):
    data_dirs = os.listdir(d)
    for directory in data_dirs:
        inputs = np.load(d+'/%s/np_arrays/inputs.npy' % directory)
        labels = np.load(d+'/%s/np_arrays/masks.npy' % directory)
        for i in range(len(inputs)):
            yield ([inputs[i]], [np.reshape(labels[i+1],(299,299,1))])

# Generators
gen = lambda: get_data(training_data_dir)
v_gen = lambda: test_data(valid_data_dir)

# TF Dataset objects (from generators)
ds = tf.data.Dataset.from_generator(gen, (tf.float32, tf.float32), output_shapes=(tf.TensorShape([299,299,3]),tf.TensorShape([299,299]))).repeat().shuffle(1000).map(augment).batch(batch_size)
v_ds = tf.data.Dataset.from_generator(v_gen, (tf.float32, tf.float32), output_shapes=(tf.TensorShape([None,299,299,3]),tf.TensorShape([None,299,299,1])))


# Reset graph
tf.reset_default_graph()

# Start TF session
with tf.Session() as sess:
    # Load the model
    saver = tf.train.import_meta_graph(os.path.join(os.path.dirname(os.path.abspath('__file__')),'..','obstacle-detection','models','stage3_final','model_final.meta'))
    saver.restore(sess,tf.train.latest_checkpoint(os.path.join(os.path.dirname(os.path.abspath('__file__')),'..','obstacle-detection','models','stage3_final')))
    print('Model restored')
    graph = tf.get_default_graph()

    # Load the tensors defined in the model
    iterator_init = graph.get_operation_by_name('iterator_init_op')
    valid_init = graph.get_operation_by_name('valid_init_op')
    train_step = graph.get_operation_by_name('Fine_tuning/train_op')
    mask = graph.get_tensor_by_name('Fine_tuning/Th_prediction:0')
    pred = graph.get_tensor_by_name('Fine_tuning/Prediction:0')
    c_loss = graph.get_tensor_by_name('Fine_tuning/c_Loss:0')
    d_loss = graph.get_tensor_by_name('Fine_tuning/d_Loss:0')
    loss = graph.get_tensor_by_name('Fine_tuning/Loss:0')
    labels = graph.get_tensor_by_name('Labels:0')    
    loss_weight = graph.get_tensor_by_name('Fine_tuning/Loss_weight:0')
    inputs = graph.get_tensor_by_name('Inputs:0')
    avg_valid = graph.get_tensor_by_name('avg_valid:0')

    # Create file writer for statistics and saver for saving the model
    writer = tf.summary.FileWriter('fine_tuning_result')
    writer.add_graph(tf.get_default_graph())
    saver2 = tf.train.Saver()

    # Create summaries to save training and validation statistics
    c_loss_summary = tf.summary.scalar('c_loss', tf.reshape(c_loss,[]))
    d_loss_summary = tf.summary.scalar('d_loss', tf.reshape(d_loss,[]))
    loss_summary = tf.summary.scalar('loss', tf.reshape(loss,[]))
    valid_summary = tf.summary.scalar('valid', avg_valid)

    # Initialize dataset's iterator
    sess.run(iterator_init)

    # Training loop
    for i in range(iterations):
        # Make predictions for a batch, update weights, compute losses and statistics
        inp, lab, p, m, _, l, cl_sum, dl_sum, l_sum = sess.run([inputs, labels, pred, mask, train_step, loss, c_loss_summary, d_loss_summary, loss_summary], feed_dict={loss_weight: 0})
        
        # Save statistics
        writer.add_summary(cl_sum,i)
        writer.add_summary(dl_sum,i)
        writer.add_summary(l_sum,i)
        print('batch:', i, 'loss:', l)

        # Do validation and save the model regularly
        if (i%saves) == 0 or i == iterations-1:
            print('Saving model and validating')
            saver2.save(sess,'fine_tuning_result/model_%s' % i)  # Saving (intermediate) model
            sess.run(valid_init)  # Initializing validation dataset
            numvalid = 0
            valid_loss = 0
            try:
                while True:
                    valid_loss += sess.run(loss, feed_dict={loss_weight: 0})  # Accumulated loss for validation dataset
                    numvalid += 1
            except tf.errors.OutOfRangeError:
                print('Validation finished')
                valid_loss = valid_loss/numvalid
                v_summary = sess.run(valid_summary, feed_dict={avg_valid: valid_loss})  # Average validation loss as statistics
                writer.add_summary(v_summary,i)
                sess.run(iterator_init)  # Re-initialize training dataset iterator
                pass
    
    # Save the final version of the model when the training is finished
    saver2.save(sess,'fine_tuning_result/model_final')