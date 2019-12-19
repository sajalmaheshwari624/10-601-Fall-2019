import sys
import numpy as np
import csv


def initWCalc(hmmPrior, hmmEmit, inputObsIndex) :
	numStates = hmmPrior.size
	wInit = np.zeros((numStates,1))
	for index,prior in enumerate(hmmPrior) :
		#print (hmmEmit[index][inputObsIndex], prior)
		wInit[index] = np.log(prior) + np.log((hmmEmit[index][inputObsIndex]))
	return wInit

def initPCalc(stateValues) :
	numStates = len(stateValues)
	pInit = np.zeros((numStates,1))
	for index, state in enumerate(stateValues) :
		pInit[index] = stateValues.index(state)
	return pInit

def WAndPCalc(hmmEmit, hmmTrans, inputObsIndex, prevWVec) :
	numStates = prevWVec.size
	wCurrent = np.zeros((numStates,1))
	currentP = np.zeros((numStates,1))
	for stateIndex in range(numStates) :
		constTerm = hmmEmit[stateIndex][inputObsIndex]
		wCurrentIndex = 0
		pCurrentIndex = 0
		wCurrentValArr = np.zeros(numStates)
		for varIndex in range(numStates) :
			wCalculated = np.log(hmmTrans[varIndex][stateIndex]) + prevWVec[varIndex] + np.log(constTerm)
			wCurrentValArr[varIndex] = wCalculated
		wCurrent[stateIndex] = np.max(wCurrentValArr)
		currentP[stateIndex] = np.argmax(wCurrentValArr)
	return wCurrent, currentP

def stateList(wList, pList) :
	pFinalSeq = []
	wLast = wList[:,-1]
	lastP = np.argmax(wLast)
	pFinalSeq.append(lastP)
	#print (pList)
	for index in range(0,pList.shape[1]-1) :
		pLast = pList[:,pList.shape[1]-index-1]
		pFinalSeq.append(int(pLast[pFinalSeq[-1]]))
	pFinalSeq.reverse()
	return pFinalSeq

def writeOutputToFile(outputLists, outFileName) :
	numLines = len(outputLists)
	with open(outFileName, 'w') as outFile :
		for i in range(numLines) :
			outList = outputLists[i]
			listLen = len(outList)
			for j in range(listLen) :
				outFile.write(outList[j])
				if j != listLen - 1 :
					outFile.write(" ")
			outFile.write('\n')


def testFileToMatrix(inFileSeq, word_index_file, state_index_file, hmmPrior, hmmEmit, hmmTrans) :

	with open(inFileSeq, 'r') as inFile :
		seqList = inFile.read().splitlines()

	with open(word_index_file, 'r') as wordFile :
		wordValues = wordFile.read().splitlines()

	with open(state_index_file, 'r') as stateFile :
		stateValues = stateFile.read().splitlines()

	accuCount = 0
	totalCount = 0
	outputLists = []
	for seq in seqList :
		wList = np.array([[],[]])
		pList = np.array([[],[]])
		eleList = seq.split(' ')
		#print (eleList)
		for index in range(len(eleList)) :
			ele = eleList[index]
			obs, state = ele.split('_')
			wordIndexCurrent = wordValues.index(obs)
			stateIndexCurrent = stateValues.index(state)
			if index == 0 :
				wList = initWCalc(hmmPrior, hmmEmit, wordIndexCurrent)
				pList = initPCalc(stateValues)
			else :
				wVec, pVec = WAndPCalc(hmmEmit, hmmTrans, wordIndexCurrent, wList[:,-1])
				wList = np.hstack((wList, wVec))
				pList = np.hstack((pList, pVec))
		stateSeq = stateList(wList, pList)
		newLen = 0
		outList = []
		for index in range(len(eleList)) :
			totalCount += 1
			ele = eleList[index]
			obs, state = ele.split('_')
			stateIndexGT = stateValues.index(state)
			if stateIndexGT == stateSeq[index] :
				accuCount += 1
			stateValPred = stateValues[stateSeq[index]]
			stringOut = obs + '_' + stateValPred
			outList.append(stringOut)
		outputLists.append(outList)

	accuracy = accuCount / totalCount
	return outputLists, accuracy




hmmPrior= np.loadtxt(sys.argv[4], dtype='float')
hmmEmit = np.loadtxt(sys.argv[5], dtype='float', delimiter = " ")
hmmTrans = np.loadtxt(sys.argv[6], dtype='float', delimiter = " ")
outputLists, accuracy = testFileToMatrix(sys.argv[1], sys.argv[2], sys.argv[3], hmmPrior, hmmEmit, hmmTrans)
writeOutputToFile(outputLists, sys.argv[7])
with open(sys.argv[8], 'w') as accuFile :
	accuFile.write("Accuracy: ")
	accuFile.write(str(accuracy))
	accuFile.write('\n')