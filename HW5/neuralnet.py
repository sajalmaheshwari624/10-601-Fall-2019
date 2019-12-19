import numpy as np
import sys
import csv
import matplotlib.pyplot as plt
import time

def getFeaturesFromFile (inFileName) :
	with open(inFileName, 'r') as inFile :
		featureList = inFile.readlines()
	tsvreader = csv.reader(featureList, delimiter=",")
	inFile.close()
	dataPoints = np.array(list(tsvreader))
	labels = dataPoints[:,0].astype('float')
	features = dataPoints[:,1:].astype('int')
	return labels, features

def sigmoid (inputs) :
	return 1 / (1 + np.exp(-inputs))


def loss (trueLabel, predictedOutput) :
	trueLabel = int (trueLabel)
	lossVal = -1*np.log(predictedOutput[trueLabel])
	return lossVal


def feedForward(feature, alphaMatrix, betaMatrix) :
	feature = np.append([1], feature, axis = 0)
	feature = np.reshape(feature, [129,1])
	hiddenLayerInput = np.matmul(alphaMatrix, feature)
	hiddenLayerOutput = sigmoid(hiddenLayerInput)
	bufferArray = np.ones([1,1])
	hiddenLayerOutput = np.append(bufferArray, hiddenLayerOutput, axis = 0)
	outputLayerInput = np.matmul(betaMatrix, hiddenLayerOutput)
	outputLayerSoftMax = np.exp(outputLayerInput)
	outputLayerOutput = outputLayerSoftMax / (np.sum(outputLayerSoftMax))
	return hiddenLayerOutput, outputLayerOutput

def backPropogation(outputValue, trueLabel, numClasses, alphaMatrix, betaMatrix, hiddenLayerPred, inputFeatures) :
	groundTruthValue = np.zeros([numClasses,1])
	trueLabel = int(trueLabel)
	groundTruthValue[trueLabel] = 1
	deltaB = outputValue - groundTruthValue
	hiddenLayerTr = np.transpose(hiddenLayerPred)
	deltaBeta = np.matmul(deltaB, hiddenLayerTr)
	betaStarMatrix = betaMatrix[:,1:]
	deltaBTr = np.transpose(deltaB)
	deltaZ = np.matmul(deltaBTr, betaStarMatrix)
	deltaA = np.transpose(deltaZ) * (hiddenLayerPred[1:]) * (1 - (hiddenLayerPred[1:]))
	inputFeatures = np.append([1.], inputFeatures, axis = 0)
	inputFeatures = np.reshape(inputFeatures, [129,1])
	deltaAlpha = np.matmul(deltaA, np.transpose(inputFeatures))
	return deltaAlpha, deltaBeta


def saveLabelFiles(labels, fileName) :
	with open(fileName, 'w') as outFile :
		for label in labels :
			outFile.write(str(int(label)))
			outFile.write('\n')

def saveMetricFile(trainLossPerEpoch, testLossPerEpoch, trainFalse, testFalse, fileName) :
	num_epochs = trainLossPerEpoch.shape[0]
	with open(fileName, 'w') as inFile :
		for i in range(num_epochs) :
			inFile.write("epoch=")
			inFile.write(str(i+1))
			inFile.write(" crossentropy(train): ")
			inFile.write(str(trainLossPerEpoch[i]))
			inFile.write('\n')
			inFile.write("epoch=")
			inFile.write(str(i+1))
			inFile.write(" crossentropy(test): ")
			inFile.write(str(testLossPerEpoch[i]))
			inFile.write('\n')

	with open(fileName, 'a') as inFile :
		inFile.write("error(train): ")
		inFile.write(str(trainFalse))
		inFile.write('\n')

		inFile.write("error(test): ")
		inFile.write(str(testFalse))
		inFile.write('\n')

def trainAndTest(trainFile, testFile, num_epoch, hiddenUnits, initializeMethod, learning_rate) :
	trainLabels, trainFeatures = getFeaturesFromFile(trainFile)
	testLabels, testFeatures = getFeaturesFromFile(testFile)
	outputClasses = 10
	featureSize = trainFeatures.shape[1]
	alphaBias = np.zeros([hiddenUnits,1])
	betaBias = np.zeros([outputClasses,1])
	if (initializeMethod == 1) :
		alphaWeights = np.random.uniform(-0.1,0.1,[hiddenUnits, featureSize])
		betaWeights = np.random.uniform(-0.1,0.1,[outputClasses,hiddenUnits])
		betaMatrix = np.append(betaBias, betaWeights, axis = 1)
		alphaMatrix = np.append(alphaBias, alphaWeights, axis = 1)
	else :
		alphaWeights = np.zeros([hiddenUnits, featureSize])
		betaWeights = np.zeros([outputClasses, hiddenUnits])
		betaMatrix = np.append(betaBias, betaWeights, axis = 1)
		alphaMatrix = np.append(alphaBias, alphaWeights, axis = 1)
	trainLossPerEpoch = np.array([])
	testLossPerEpoch = np.array([])
	for epoch in range(num_epoch) :
		count = 0
		for feature in trainFeatures :
			trueLabel = trainLabels[count]
			count += 1
			hiddenLayerPred, labelPred = feedForward(feature, alphaMatrix, betaMatrix)
			deltaAlpha, deltaBeta = backPropogation(labelPred, trueLabel, outputClasses, alphaMatrix, betaMatrix, hiddenLayerPred, feature)
			alphaMatrix = alphaMatrix - learning_rate*deltaAlpha
			betaMatrix = betaMatrix - learning_rate*deltaBeta

		count = 0
		trainLoss = np.array([])
		lossValTrain = 0
		for feature in trainFeatures :
			trueLabel = trainLabels[count]
			count += 1
			hiddenLayerPred, labelPred = feedForward(feature, alphaMatrix, betaMatrix)
			lossValTrain += loss(trueLabel, labelPred)
		lossValTrain = float((lossValTrain/count))
		trainLossPerEpoch = np.append(trainLossPerEpoch, lossValTrain)

		count = 0
		testLoss = np.array([])
		lossValTest = 0
		for feature in testFeatures :
			trueLabel = testLabels[count]
			count += 1
			hiddenLayerPred, labelPred = feedForward(feature, alphaMatrix, betaMatrix)
			lossValTest += float(loss(trueLabel, labelPred))
		lossValTest = (lossValTest/count)
		testLossPerEpoch = np.append(testLossPerEpoch, lossValTest)

		'''
		with open(metricFileName, 'a+') as metricFile :
			metricFile.write("epoch=")
			metricFile.write(str(epoch+1))
			metricFile.write(" crossentropy(train): ")
			metricFile.write(str(lossValTrain))
			metricFile.write('\n')
			metricFile.write("epoch=")
			metricFile.write(str(epoch+1))
			metricFile.write(" crossentropy(test): ")
			metricFile.write(str(lossValTest))
			metricFile.write('\n')
		'''

	trainFalse = 0
	count  = 0
	outputLabelTrain = np.array([])
	for feature in trainFeatures :
		trueLabel = trainLabels[count]
		count += 1
		_, labelPred = feedForward(feature, alphaMatrix, betaMatrix)
		labelPred = np.argmax(labelPred)
		outputLabelTrain = np.append(outputLabelTrain, labelPred)
		if trueLabel != labelPred :
			trainFalse += 1
	trainFalse = trainFalse/count
	testFalse = 0
	count = 0
	outputLabelTest = np.array([])
	for feature in testFeatures :
		trueLabel = testLabels[count]
		count += 1
		_, labelPred = feedForward(feature, alphaMatrix, betaMatrix)
		labelPred = np.argmax(labelPred)
		outputLabelTest = np.append(outputLabelTest, labelPred)
		#print (labelPred)
		if trueLabel != labelPred :
			testFalse += 1
	testFalse = testFalse/count

	'''
	with open(metricFileName, 'a+') as metricFile :
		metricFile.write("error(train): ")
		metricFile.write(str(trainFalse))
		metricFile.write('\n')

		metricFile.write("error(test): ")
		metricFile.write(str(testFalse))
		metricFile.write('\n')
	'''

	return outputLabelTrain, outputLabelTest, trainLossPerEpoch, testLossPerEpoch, trainFalse, testFalse


if __name__ == "__main__" :
	trainFile = sys.argv[1]
	testFile = sys.argv[2]
	trainOutFile = sys.argv[3]
	testOutFile = sys.argv[4]
	metricFile = sys.argv[5]
	num_epoch = int(sys.argv[6])
	hiddenUnits = int(sys.argv[7])
	initFlag = int(sys.argv[8])
	learning_rate = float(sys.argv[9])
	outputLabelTrain, outputLabelTest, trainLossPerEpoch, testLossPerEpoch, trainFalse, testFalse = trainAndTest(trainFile, testFile, num_epoch, hiddenUnits, initFlag, learning_rate)
	#print (outputLabelTest)
	saveLabelFiles(outputLabelTrain, trainOutFile)
	saveLabelFiles(outputLabelTest, testOutFile)
	saveMetricFile(trainLossPerEpoch, testLossPerEpoch, trainFalse, testFalse, metricFile)