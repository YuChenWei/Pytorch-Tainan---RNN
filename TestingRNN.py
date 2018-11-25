import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import CSV_Reader as csvR
import math
from sklearn.preprocessing import StandardScaler as scaler

FileReader = csvR.CSVReader()

# Training parameter
stateSize = 4
numEpochs = 800
numTimeSteps = 1
numFeature = 1
numTarget = 1
learningRate = 0.1

# Reset computing graph
tf.reset_default_graph()

# Creating placeholder for feature and target
FeatureP = tf.placeholder(tf.float32, [None, numTimeSteps, numFeature], name = 'FeatureP')
TargetP = tf.placeholder(tf.float32, [None, numTimeSteps, numTarget], name = 'TargetP')

FeatureInput = tf.unstack(FeatureP, axis = 1)


#cell = tf.nn.rnn_cell.BasicRNNCell(stateSize)
#cell = tf.nn.rnn_cell.LSTMCell(stateSize)
cell = tf.nn.rnn_cell.GRUCell(stateSize)

# Using static RNN
rnnOuts, FinalState = tf.nn.static_rnn(cell,
                                       FeatureInput,
                                       dtype = tf.float32)

# Creating variable for weighting and bias
W = tf.get_variable('W', [stateSize, numTarget])
b = tf.get_variable('b', [numTarget], initializer = tf.constant_initializer(0.0))

predictions = [tf.matmul(rnnOut, W)+b for rnnOut in rnnOuts]
print(predictions)

targetAsList = tf.unstack(TargetP, num = numTimeSteps, axis = 1)
print(targetAsList)

# Loss function using mean square error
MSE = tf.losses.mean_squared_error
Losses = [MSE(labels = label, predictions = prediction) for prediction, label in zip(predictions, targetAsList)]

totalLoss = tf.reduce_mean(Losses)
Optimizer = tf.train.AdagradOptimizer(learning_rate = learningRate).minimize(totalLoss)

with tf.Session() as tfs:
    tfs.run(tf.global_variables_initializer())
    epochLoss = 0.0
    for epoch in range(numEpochs):
        # Using dictionaries for feeding feature and target
        FeedDic = {FeatureP: FileReader.TrainFeature.reshape(-1, numTimeSteps, 1),
                   TargetP: FileReader.TrainTarget.reshape(-1, numTimeSteps, 1)}
        epochLoss, TrainPred, _ = tfs.run([totalLoss, predictions, Optimizer], feed_dict = FeedDic)
        
    print('Train MSE = {0}'.format(epochLoss))
    
    FeedDic = {FeatureP: FileReader.TestFeature.reshape(-1, numTimeSteps, 1),
               TargetP: FileReader.TestTarget.reshape(-1, numTimeSteps, 1)}
    
    testLoss, TestPred = tfs.run([totalLoss, predictions], feed_dict = FeedDic)
    
    print('Test MSE = {0}'.format(testLoss))
    print('Test RMSE = {0}'.format(math.sqrt(testLoss)))
    
TargetTrainPred = TrainPred[0]
TargetTestPred = TestPred[0]

# Convert the normalized data back to no-normalized data
TargetTrainPred = FileReader.scaler.inverse_transform(TargetTrainPred)
TargetTestPred = FileReader.scaler.inverse_transform(TargetTestPred)

TargetTrainOri = FileReader.scaler.inverse_transform(FileReader.TrainTarget)
TargetTestOri = FileReader.scaler.inverse_transform(FileReader.TestTarget)

Diagram = plt.figure()
plt.plot(np.arange(len(FileReader.Dataset)), FileReader.Dataset, 'b', label = 'Original')
plt.plot(np.arange(len(TargetTrainPred)), TargetTrainPred, 'r--', label = 'Train')
plt.plot(np.arange(len(TargetTestPred)) + len(TargetTrainPred)-1, TargetTestPred, 'g:', label = 'Test')
plt.legend(loc='upper left')
print(np.arange(len(TargetTestPred)) + len(TargetTrainPred)-1)

plt.show()
Diagram.savefig('Result.png')





