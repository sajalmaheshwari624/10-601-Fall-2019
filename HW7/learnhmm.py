import sys
import numpy as np
import csv

def getHMMMatrices(inFileSeq, word_index_file, state_index_file) :
	with open(inFileSeq, 'r') as inFile :
		seqList = inFile.read().splitlines()

	with open(word_index_file, 'r') as wordFile :
		wordIndices = wordFile.read().splitlines()

	with open(state_index_file, 'r') as stateFile :
		stateIndices = stateFile.read().splitlines()

	numStates = len(stateIndices)
	numObs = len(wordIndices)
	hmmTrans = np.ones((numStates, numStates))
	hmmEmit = np.ones((numStates, numObs))
	hmmPrior = np.ones(numStates)	
	for seq in seqList :
		eleList = seq.split(' ')
		if len(eleList) >= 1 :
			for index in range(len(eleList)-1) :
				ele = eleList[index]
				obs, state = ele.split('_')
				wordIndexCurrent = wordIndices.index(obs)
				stateIndexCurrent = stateIndices.index(state)

				eleNext = eleList[index + 1]
				obsNext, stateNext = eleNext.split('_')
				wordIndexNext = wordIndices.index(obsNext)
				stateIndexNext = stateIndices.index(stateNext)
				hmmTrans[stateIndexCurrent][stateIndexNext] += 1.

				hmmEmit[stateIndexCurrent][wordIndexCurrent] += 1.

			lastEle = eleList[-1]
			obs, state = lastEle.split('_')
			wordIndexCurrent = wordIndices.index(obs)
			stateIndexCurrent = stateIndices.index(state)
			hmmEmit[stateIndexCurrent][wordIndexCurrent] += 1.

		firstEle = eleList[0]
		_, state = firstEle.split('_')
		stateIndex = stateIndices.index(state)
		hmmPrior[stateIndex] += 1

	hmmPrior = hmmPrior / np.sum(hmmPrior)
	hmmTransSum = hmmTrans.sum(axis=1, keepdims=True)
	hmmTrans = hmmTrans / hmmTransSum
	hmmEmitSum = hmmEmit.sum(axis=1, keepdims=True)
	hmmEmit = hmmEmit / hmmEmitSum

	return hmmTrans, hmmPrior, hmmEmit

def writeMatrixToFile(matrix, outFileName) :
	with open(outFileName, 'w') as outFile :
		for rows in range(matrix.shape[0]) :
			for cols in range(matrix.shape[1]) :
				element = matrix[rows][cols]
				outFile.write(str(element))
				if cols != matrix.shape[1] - 1 :
					outFile.write(" ")
			outFile.write('\n')

def writeVectorToFile(vector, outFileName) :
	with open(outFileName, 'w') as outFile :
		for index in range(vector.size) :
			element = vector[index]
			outFile.write(str(element))
			outFile.write('\n')


hmmTrans, hmmPrior, hmmEmit = getHMMMatrices(sys.argv[1], sys.argv[2], sys.argv[3])
writeMatrixToFile(hmmTrans, sys.argv[6])
writeMatrixToFile(hmmEmit, sys.argv[5])
writeVectorToFile(hmmPrior, sys.argv[4])