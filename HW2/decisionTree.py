import numpy as np
import scipy
import sys
import csv

class TreeNode :
	def __init__(self, data, labels, depth, parentVisited, parent = -1, splitAttribute = -1, error = 0, zeroElement = 0) :
		self.data = data
		self.labels = labels
		self.depth = depth
		self.error = error
		self.parentVisited = parentVisited
		self.splitAttribute = splitAttribute
		self.parent = parent
		self.guess = None
		self.minorityElement = None
		self.leftNode = None
		self.rightNode = None

def logBase2(num) :
	if num == 0 :
		return 0
	else :
		return np.log2(num)

def entropy(binaryList) :
	if len(binaryList) == 0 :
		return 0
	uniqueElementCount = np.zeros(2)
	if len(uniqueElementCount) == 0 :
		return 0
	element1 = binaryList[0]
	for i in range(len(binaryList)) :
		if binaryList[i] == element1 :
			uniqueElementCount[0] += 1
		else :
			uniqueElementCount[1] += 1
	zeroProbability = float(uniqueElementCount[0]) / np.sum(uniqueElementCount)
	oneProbability = float(uniqueElementCount[1]) / np.sum(uniqueElementCount)

	return -1 * zeroProbability * logBase2(zeroProbability) - oneProbability * logBase2(oneProbability)

def majorityElement(binaryList) :
	uniqueElementCount = np.zeros(2)
	if len(binaryList) == 0 :
		return uniqueElementCount
	elif len(binaryList) == 1 :
		return [binaryList[0], -1]
	element1 = binaryList[0]
	countElement1 = binaryList.count(element1)
	countElement2 = len(binaryList) - countElement1
	if (countElement1 >= countElement2) :
		majorElement = element1
		minorElement = element1
		count = 0
		for i in range(len(binaryList)) :
			if binaryList[i] != majorElement :
				count = count + 1
				minorElement = binaryList[i]
		if count == 0 :
			return [majorElement, -1]
		return [majorElement, minorElement]
	else :
		for i in range(len(binaryList)) :
			if binaryList[i] != element1 :
				element2 = binaryList[i]
				return [element2, element1]

def majorityClassifier(binaryList) :
	majorElement = majorityElement(binaryList)
	majorityCount = binaryList.count(majorElement)
	minorityCount = len(binaryList) - majorityCount
	if len(binaryList) == 0 :
		return 0
	return float(minorityCount) / len(binaryList)


def mutualInfo(binaryList1, binaryList2) :
	uniqueElementCountB1 = np.zeros(2)
	entropyList2 = entropy(binaryList2)
	#print (entropyList2)
	element1 = binaryList1[0]
	zeroElementIndices = []
	oneElementIndices = []

	for i in range(len(binaryList1)) :
		if binaryList1[i] == element1 :
			uniqueElementCountB1[0] += 1
			zeroElementIndices.append(i)
		else :
			uniqueElementCountB1[1] += 1
			oneElementIndices.append(i)

	binaryList2ZeroBin = []
	binaryList2OneBin = []
	for i in range(len(zeroElementIndices)) :
		binaryList2ZeroBin.append(binaryList2[zeroElementIndices[i]])

	for i in range(len(oneElementIndices)) :
		binaryList2OneBin.append(binaryList2[oneElementIndices[i]])

	binaryListZeroBinEntropy = entropy(binaryList2ZeroBin)
	binaryListOneBinEntropy = entropy(binaryList2OneBin)

	zeroProbabilityList1 = float(len(zeroElementIndices)) / len(binaryList1)
	oneProbabilityList1 = float(len(oneElementIndices)) / len(binaryList1)

	conditionalEntropy = zeroProbabilityList1 * binaryListZeroBinEntropy + oneProbabilityList1 * binaryListOneBinEntropy
	return entropyList2 - conditionalEntropy

def trainTree(root, maxDepth) :
	if TreeNode:
		#print (root.parentVisited)
		[defaultGuess,minorityElement] = majorityElement(root.labels)
		root.guess = defaultGuess
		root.minorityElement = minorityElement
		if (minorityElement == -1) :
			root.minorityElement = "Only majority attribute present!"
		root.error = majorityClassifier(root.labels)
		if entropy(root.labels) == 0 :
			return root
		isEveryNodeVisited = np.min(root.parentVisited)
		#print (len(root.data), root.parentVisited, root.guess, root.error)
		if isEveryNodeVisited == 1 :
			return root
		if (root.depth >= maxDepth) :
			return root
		else :
			attributeCount = len(root.data[0]) - 1
			attributeMutualInfo = np.zeros(attributeCount)
			maxMutualInfoAttribute = 0
			maxMutualInfo = 0
			for i  in range(attributeCount) :
				if root.parentVisited[i] == 0 :
					attributeData = [row[i] for row in root.data]
					attributeMutualInfo[i] = mutualInfo(root.labels, attributeData)
					if attributeMutualInfo[i] > maxMutualInfo :
						maxMutualInfo = attributeMutualInfo[i]
						maxMutualInfoAttribute = i
			visitedAttributes = np.zeros(len(root.parentVisited))
			for i in range(attributeCount) :
				visitedAttributes[i] = root.parentVisited[i]
			visitedAttributes[maxMutualInfoAttribute] = 1
			root.splitAttribute = maxMutualInfoAttribute
			attributeCol = [row[maxMutualInfoAttribute] for row in root.data]
			zeroElement = attributeCol[0]
			root.zeroElement = zeroElement
			dataLeft = []
			dataRight = []
			labelLeft = []
			labelRight = []
			for j in range(len(attributeCol)) :
				if attributeCol[j] == zeroElement :
					dataLeft.append(root.data[j])
					#print (len(root.data[j]))
					labelLeft.append(root.labels[j])
				else :
					dataRight.append(root.data[j])
					labelRight.append(root.labels[j])
			root.leftNode = TreeNode(dataLeft, labelLeft, root.depth + 1, visitedAttributes, maxMutualInfoAttribute)
			root.rightNode = TreeNode(dataRight, labelRight, root.depth + 1, visitedAttributes, maxMutualInfoAttribute)
			trainTree(root.leftNode, maxDepth)
			trainTree(root.rightNode, maxDepth)

def testTree(root, testVector, outputLabels) :
	if (root.leftNode == None and root.rightNode == None) :
		outputLabels.append(root.guess)
	else :
		partitionAttribute = root.splitAttribute
		if (testVector[partitionAttribute] == root.zeroElement) :
			#print (testVector[partitionAttribute], root.zeroElement)
			testTree(root.leftNode, testVector, outputLabels)
		else :
			#print (testVector[partitionAttribute], root.zeroElement)
			testTree(root.rightNode, testVector, outputLabels)

def inputFileToData(fileName) :
	with open(fileName) as dataFile :
		tsvreader = csv.reader(dataFile, delimiter="\t")
		dataMatrix = list(list(tsvreader))
	attributeNames = dataMatrix[0]
	dataMatrix = dataMatrix[1:]
	#print (attributeNames)
	dataLabels = [row[-1] for row in dataMatrix]

	return (dataMatrix, dataLabels, attributeNames)

def outputLabelToFile(outputs, fileName) :
	outFile = open(fileName,"w+")
	num = len(outputs)
	for i in range(num - 1) :
		outFile.write(outputs[i])
		outFile.write('\n')
	outFile.write(outputs[num - 1])

def printTree(root, attributeNames) :
	if root :
		Guess = root.labels.count(root.guess)
		notGuess = len(root.labels) - Guess
		if root.parent == -1 :
			print(root.guess + " " + str(Guess) +  " " + root.minorityElement + " " + str(notGuess))
			printTree(root.leftNode, attributeNames)
			printTree(root.rightNode, attributeNames)
		else :
			print((root.depth + 1) * " " + str(attributeNames[root.parent]) + " = "  + root.guess + " " + str(Guess) +  " " + root.minorityElement + " " + str(notGuess))
			printTree(root.leftNode, attributeNames)
			printTree(root.rightNode, attributeNames)

trainDataFile = sys.argv[1]
testDataFile = sys.argv[2]
[trainDataMatrix, trainLabels, attributeNames] = inputFileToData(trainDataFile)
#print (trainLabels)
[testDataMatrix, testLabels, attributeNames] = inputFileToData(testDataFile)

attributeCount = len(trainDataMatrix[0]) - 1
attributes = np.arange(attributeCount)
visitedAttributesTrain = np.zeros(attributeCount)
maxDepth = int(sys.argv[3])

root = TreeNode(trainDataMatrix, trainLabels, 0, visitedAttributesTrain)
trainTree(root, maxDepth)
#printTree(root, attributeNames)

outputLabelsTrain = []
for i in range(len(trainDataMatrix)) :
	testTree(root, trainDataMatrix[i], outputLabelsTrain)

count = 0
countIndex = 0
for i in range(len(trainLabels)) :
	countIndex += 1
	if (trainLabels[i] == outputLabelsTrain[i]) :
		count += 1
errorTrain = 1 - float(count) / len(trainLabels)

outFileNameTrain = sys.argv[4]
outputLabelToFile(outputLabelsTrain, outFileNameTrain)

outputLabelsTest = []
for i in range(len(testDataMatrix)) :
	testTree(root, testDataMatrix[i], outputLabelsTest)

count = 0
for i in range(len(testLabels)) :
	if (testLabels[i] == outputLabelsTest[i]) :
		count += 1
errorTest = 1 - float(count) / len(testLabels)

outFileNameTest = sys.argv[5]
outputLabelToFile(outputLabelsTest, outFileNameTest)

metricFileName = open(sys.argv[6],"w+")
metricFileName.write("error(train): " + str(errorTrain) + '\n')
metricFileName.write("error(test): " + str(errorTest) + '\n')

Y = [0,1,1,1,0,0,0]
B = [1,1,0,1,0,1,0]

print (mutualInfo(Y,B))


