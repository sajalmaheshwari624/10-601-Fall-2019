import numpy as np
import scipy as sp
import sys
import csv

def MajorityVoteClassifier(binaryCountList) :
	totalSum = np.sum(binaryCountList)
	minValue = min(binaryCountList)
	return float(minValue) / totalSum

def CalculateEntropy(binaryCountList) :
	totalSum = np.sum(binaryCountList)
	zeroProbability = float(binaryCountList[0]) / totalSum
	oneProbability = float(binaryCountList[1]) / totalSum

	return -1 * zeroProbability * np.log2(zeroProbability) - oneProbability * np.log2(oneProbability)

binaryCountList = [0, 0]
oneString = ""
inFilename = sys.argv[1]
outFileName= open(sys.argv[2],"w+")
with open(inFilename) as inFile :
	tsvreader = csv.reader(inFile, delimiter="\t")
	count = 0
	for dataPoint in tsvreader :
		if count == 0 :
			pass
		elif count == 1 :
			oneString = dataPoint[-1]
			binaryCountList[1] += 1
		else :
			if dataPoint[-1] == oneString :
				binaryCountList[1] += 1
			else :
				binaryCountList[0] += 1
		count += 1

majorityError = MajorityVoteClassifier(binaryCountList)
entropy = CalculateEntropy(binaryCountList)

outFileName.write("entropy: " + str(entropy) + '\n')
outFileName.write("error: " + str(majorityError) + '\n')