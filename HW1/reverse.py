import sys
filename = sys.argv[1]
fileList = open(filename).readlines()
fileList.reverse()
outFileName= open(sys.argv[2],"w+")
num = len(fileList)
for i in range(num) :
	#print (fileList[i])
	outFileName.write(fileList[i])