import numpy as np
import sys
import csv
import matplotlib.pyplot as plt
import time


def getDataFromFile(inFileName) :
	with open(inFileName, 'r') as inFile :
		fileList = inFile.readlines()
	tsvreader = csv.reader(fileList, delimiter="\t")
	inFile.close()
	dataPoints = list(tsvreader)
	labelVec = np.array([])
	dataList = []
	for rows in dataPoints :
		#print (rows)
		dataDict = {}
		label = int(rows[0])
		labelVec = np.append(labelVec, label)
		data = rows[1:]
		#print (data)
		for lists in data :
			#print (lists)
			index, val = lists.split(':')
			index = int(index) + 1
			val = int(val)
			dataDict[index] = val
		#print (dataDict)
		#print ("======================================")
		dataDict[0] = 1
		#print (dataDict)
		dataList.append(dataDict)
		#print (dataList)
	return labelVec, dataList


def SGD(truelabel, predProb, inputExample, currentWeight, lr) :
	scalarFactor = truelabel - predProb
	#print (scalarFactor)
	for key, val in inputExample.items() :
		currentWeight[key] = currentWeight[key] + lr*scalarFactor*val
	return currentWeight

def getError(trueLabels, dataDictionary, weights) :
	accurateCount = 0
	num_examples = len(trueLabels)
	#print (num_examples)
	for num, example in enumerate(dataDictionary) :
		labelNum = trueLabels[num]
		product = 0
		#print (weights[0])
		for key, val in example.items() :
			product += weights[key]
		predProb = np.divide(1., 1 + np.exp(-1*product))
		#print (predProb, labelNum)
		if predProb >= 0.5 and labelNum == 1 :
			accurateCount += 1
		elif predProb < 0.5 and labelNum == 0 :
			accurateCount += 1
	error = 1 - (accurateCount / num_examples)
	return error
		
def test(testInFile, weights) :
	accurateCount = 0
	predLabels = []
	truelabels, testDictionary = getDataFromFile(testInFile)
	num_examples = len(truelabels)
	for num, example in enumerate(testDictionary) :
		labelNum = truelabels[num]
		product = 0
		for key, val in example.items() :
			product += weights[key]
		predProb = np.divide(1, 1 + np.exp(-1*product))
		#print (product)
		if predProb >= 0.5 :
			predLabel =1
			predLabels.append('1')
			#predLabels = predLabels + '1'
		else :
			predLabel = 0
			predLabels.append('0')
			#predLabels = predLabels + '0'
		#print (predLabel)
		if predProb >= 0.5 and labelNum == 1 :
			accurateCount += 1
		elif predProb < 0.5 and labelNum == 0 :
			accurateCount += 1
	error = 1 - (accurateCount / num_examples)
	return error, predLabels

def trainAndValidate(trainInFile, validInFile, num_epochs, dictFile) :
	dictionaryFile = open(dictFile, 'r')
	dictLine = dictionaryFile.readlines()
	dictionaryFile.close()
	weights = np.zeros(len(dictLine) + 1)
	lr = 0.1
	truelabelsValid, dataDictionaryValid = getDataFromFile(validInFile)
	#return 0,0,0
	truelabels, dataDictionary = getDataFromFile(trainInFile)
	#print (dataDictionary)
	validationError = np.array([])
	trainingError = np.array([])
	for epoch in range(num_epochs) :
		validationError = np.append(validationError, getError(truelabelsValid, dataDictionaryValid, weights))
		trainingError = np.append(trainingError, getError(truelabels, dataDictionary, weights))
		num_examples = len(truelabels)
		for num, example in enumerate(dataDictionary) :
			labelNum = truelabels[num]
			product = 0
			for key, val in example.items() :
				product += weights[key]
			predProb = np.divide(1, (1. + np.exp(-1*product)))
			#print (labelNum - predProb)
			#print (predProb, labelNum)
			weights = SGD(labelNum, predProb, example, weights, lr)

	return weights, trainingError, validationError


#labels, dataDictionary = getDataFromFile('../handout/smalldata/smalltrain_formatted.tsv')
finalWeight, trainingError, validationError = trainAndValidate(sys.argv[1], sys.argv[2], int(sys.argv[8]), sys.argv[4])
#print (finalWeight)
'''
f = open(sys.argv[9], 'w')
for w in finalWeight :
	f.write(str(w))
	f.write('\n')
'''
trainError, trainLabels = test(sys.argv[1], finalWeight)
validError, validLabels = test(sys.argv[2], finalWeight)
testError,testLabels = test(sys.argv[3], finalWeight)
#print (trainError)
#print (testError)

metricFile = open(sys.argv[7], 'w')

metricFile.write("error(train): ")
metricFile.write(str(trainError))
metricFile.write('\n')

metricFile.write("error(test): ")
metricFile.write(str(testError))
metricFile.write('\n')

metricFile.close()

with open(sys.argv[5], 'w') as outFileTrain :
	#print (trainError[count])
	for labels in trainLabels :
		outFileTrain.write(labels)
		outFileTrain.write('\n')

with open(sys.argv[6], 'w') as outFileTest :
	for labels in testLabels :
		outFileTest.write(labels)
		outFileTest.write('\n')