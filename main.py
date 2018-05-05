'''
THINGS TO CHANGE
- one shot iterator for val and test data
- variable names in the network
- placeholder for X,y from train, val and test and feed_dict instead in the sess.run
- change to take data from the actual test and train set
'''
import tensorflow as tf
import glob
import csv
import time

N_CLASSES = 10
BATCH_SIZE = 64
N_EPOCHS = 150

train_path = 'data/train'
labels_path = 'data/labels.txt'
test_path = 'data/test'


def conv_model(X, N_CLASSES, reuse, is_training):
    
    with tf.variable_scope('Conv', reuse = reuse): #to reuse weights and biases for testing
        
        input_layer = tf.reshape(X, [-1, 32,32,3])

        conv0 = tf.layers.conv2d(
          inputs=input_layer,
          filters=32,
          kernel_size=[3, 3],
          padding="same",
          activation=tf.nn.relu)
        
        batchnorm0 = tf.layers.batch_normalization(conv0)
        
        pool0 = tf.layers.max_pooling2d(
            inputs = batchnorm0,
            pool_size = 2,
            strides = 2,
            padding = "same",
            data_format='channels_last')
        
        conv1 = tf.layers.conv2d(
          inputs=pool0,
          filters=64,
          kernel_size=[3, 3],
          padding="same",
          activation=tf.nn.relu) 
       
        batchnorm1 = tf.layers.batch_normalization(conv1)

        pool1 = tf.layers.max_pooling2d(
            inputs = batchnorm1,
            pool_size = 2,
            strides = 2,
            padding = "same",
            data_format='channels_last')
          
        dropout0 = tf.layers.dropout(
            inputs = pool1,
            rate=0.30,
            training = is_training)
        
        conv2 = tf.layers.conv2d(
            inputs=dropout0,
            filters=128,
            kernel_size=[3, 3],
            padding="same",
            activation=tf.nn.relu)

        batchnorm2 = tf.layers.batch_normalization(conv2)
        
        pool2 = tf.layers.max_pooling2d(
            inputs = batchnorm2,
            pool_size = 2,
            strides = 2,
            padding="same",
            data_format='channels_last')
       
        conv3 = tf.layers.conv2d(
          inputs=pool2,
          filters=256,
          kernel_size=[3, 3],
          padding="same",
          activation=tf.nn.relu)
        
        batchnorm3 = tf.layers.batch_normalization(conv3)
        
        pool3 = tf.layers.max_pooling2d(
            inputs = batchnorm3,
            pool_size = 2,
            strides = 2,
            padding="same",
            data_format='channels_last')
        
        dropout1 = tf.layers.dropout(
            inputs = pool3,
            rate=0.25,
            training = is_training)

        flatten = tf.layers.flatten(dropout1)

        dense1 = tf.layers.dense(
            inputs = flatten,
            units = 1024,
            activation= tf.nn.relu)

        dense2 = tf.layers.dense(
            inputs = dense1,
            units = 512,
            activation= tf.nn.relu)

        dropout2 = tf.layers.dropout(
            inputs = dense2,
            rate=0.35,
            training = is_training)
        
        dense3 = tf.layers.dense(
            inputs = dropout2,
            units = N_CLASSES)
        
        if is_training: last_layer = dense3     #using sparse cross entropy so no need to apply softmax here
        else: last_layer = tf.nn.softmax(dense3)   #for inference

        return last_layer

X,y = train_batch
valX, valY = val_batch

global_step = tf.Variable(0, dtype=tf.int32, trainable = False, name='global_step')

train_output = conv_model(X, N_CLASSES, reuse = False, is_training = True)

cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits (labels = y, logits = train_output))
learning_rate = tf.placeholder(tf.float32)
optimizer = tf.train.AdamOptimizer().minimize(cost, global_step = global_step)

#NOTE: THIS IS VERY INEFFICIENT. IMPROVE THIS BY USING feed_dict

# Evaluate model with train data
test_output = conv_model(X, N_CLASSES, reuse=True, is_training=False)
correct_pred = tf.equal(tf.argmax(test_output, 1, output_type=tf.int32), y)
train_accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# # Evaluate model with Validation data
val_test_output = conv_model(valX, N_CLASSES, reuse=True, is_training=False)
val_pred = tf.equal(tf.argmax(val_test_output, 1, output_type=tf.int32), valY)
val_accuracy = tf.reduce_mean(tf.cast(val_pred, tf.float32))

#for test data
test_r = conv_model(X_t, N_CLASSES, reuse=True, is_training=False)
test_pred = tf.argmax(test_r, 1 , output_type=tf.int32)


init = tf.global_variables_initializer()
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)   #initialize variables
    sess.run(train_iterator.initializer)
    
    
    epochCount = 1
        
    while (epochCount < N_EPOCHS):
        startTime = time.time()
        while True:
            try:
                sess.run(optimizer)
                
            except tf.errors.OutOfRangeError:
                sess.run(train_iterator.initializer)
                tr_loss, tr_acc = sess.run([cost, train_accuracy])
                
                sess.run(val_iterator.initializer)
                val_acc = sess.run(val_accuracy)
                print("Epoch {}    Loss: {:,.4f}    Train Accuracy: {:,.2f}    Val Accuracy: {:,.2f}    Time: {:,.2f}"         
                      .format(epochCount, tr_loss, tr_acc, val_acc, time.time() - startTime))
                epochCount += 1
                break
                
            except KeyboardInterrupt:   #use this to close the program and save the model as well as a file for predictions on the test set
                print ('\nTraining Interrupted at epoch %d' % epochCount)
                epochCount = N_EPOCHS + 1
                break

    print ('Done Training')
    #Save the model
    save_path = saver.save(sess, 'checkpoints/', global_step = global_step )
    print ('Model saved at %s' % save_path)  
    
    sess.run(test_iterator.initializer)
    test_predictions = sess.run(test_pred)
    predictions_csv = open('model4_1_'+str(epochCount)+'.csv', 'w')
    header = ['id','label']
    with predictions_csv:
        writer = csv.writer(predictions_csv)
        writer.writerow((header[0], header[1]))
        for count, row in enumerate(range(test_predictions.shape[0])):
            writer.writerow((count, id2label[test_predictions[count]]))
        print("Writing complete")