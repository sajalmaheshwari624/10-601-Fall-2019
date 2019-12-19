import numpy as np
import sys
import csv


def inputFileToData(inFileName, outDataFileName, dictFile, featureType) :
	featureType = int(featureType)
	with open(dictFile, 'r') as dictionaryFile :
		dictLine = dictionaryFile.readlines()
		for i, rows in enumerate(dictLine) : 
			dictData = rows.split()
			if i == 0 :
				dictWords =  [dictData[0]]
				dictFreq = [dictData[1]]
			else :
				dictFreq.append(dictData[1])
				dictWords.append(dictData[0])
	dictionaryFile.close()
	outFile = open(outDataFileName, 'w')
	with open(inFileName, 'r') as inFile :
		tsvreader = csv.reader(inFile, delimiter="\t")
		dataPoints = list(tsvreader)
		if featureType == 1 :
			for rows in dataPoints :
				label = rows[0]
				data = rows[1:]
				bigDataStr = '\t'.join(data)
				data = [var.split('\t') for var in bigDataStr.split(' ')]
				#print (len(data))
				visited = np.zeros([len(dictFreq),1])
				outFile.write(label)
				outFile.write('\t')
				outDataList = []
				for item in data :
					#print (item[0])
					if item[0] in dictWords :
						dictIndex = dictWords.index(item[0])
						if visited[dictIndex] == 0 :
							#print (item[0])
							newFeature = str(dictIndex)
							newFeature = newFeature + ":"
							newFeature = newFeature + str(1)
							#outFile.write(str(dictIndex))
							#outFile.write(":")
							#outFile.write(str(1))
							#outFile.write('\t')
							outDataList.append(newFeature)
							visited[dictIndex] = 1
				stringToWrite = '\t'.join(outDataList)
				outFile.write(stringToWrite)
				outFile.write('\n')
		if featureType == 2 :
			for rows in dataPoints :
				label = rows[0]
				data = rows[1:]
				bigDataStr = '\t'.join(data)
				data = [var.split('\t') for var in bigDataStr.split(' ')]
				visited = np.zeros([len(dictFreq),1])
				outFile.write(label)
				outFile.write('\t')
				outDataList = []
				for item in data :
					if item[0] in dictWords :
						dictIndex = dictWords.index(item[0])
						#print (data.count(item))
						if data.count(item) < 4 and visited[dictIndex] == 0:
							#print (item[0])
							newFeature = str(dictIndex)
							newFeature = newFeature + ":"
							newFeature = newFeature + str(1)
							#outFile.write(str(dictIndex))
							#outFile.write(":")
							#outFile.write(str(1))
							#outFile.write('\t')
							outDataList.append(newFeature)
							visited[dictIndex] = 1
				stringToWrite = '\t'.join(outDataList)
				outFile.write(stringToWrite)
				outFile.write('\n')	
	inFile.close()
	outFile.close()

inputFileToData(sys.argv[1], sys.argv[5], sys.argv[4], sys.argv[8])
inputFileToData(sys.argv[2], sys.argv[6], sys.argv[4], sys.argv[8])
inputFileToData(sys.argv[3], sys.argv[7], sys.argv[4], sys.argv[8])