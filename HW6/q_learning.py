from environment import MountainCar
import sys
import numpy as np

def stateDictionaryToArray(stateSpace, stateDict) :
	stateArray = np.zeros(stateSpace)
	for keys,values in stateDict.items() :
		#print (keys)
		stateArray[keys] = stateDict[keys]
	return stateArray

def saveRewardFile(rewards, fileName) :
	with open(fileName, 'w') as outFile :
		for reward in rewards :
			outFile.write(str(reward))
			outFile.write('\n')

def saveWeightFile(bias, weightMatrix, fileName) :
	with open(fileName, 'w') as outFile :
		outFile.write(str(bias))
		outFile.write('\n')
		for j in range(weightMatrix.shape[1]) :
			for i in range(weightMatrix.shape[0]) :
				outFile.write(str(weightMatrix[i][j]))
				outFile.write('\n')
def main(args):
	mode = args[1]
	weightFileName = args[2]
	rewardFileName = args[3]
	num_episodes = int(args[4])
	num_maxiter = int(args[5])
	epsilon = float(args[6])
	gamma = float(args[7])
	alpha = float(args[8])

	worldEnv = MountainCar(mode)

	if mode == 'raw' :
		stateSpace = 2
	else :
		stateSpace = 2048

	numAction = 3
	weightMatrix = np.zeros((numAction, stateSpace))
	bias = 0.0
	rewardList = np.array([])
	#print (worldEnv.reset())
	for i in range(num_episodes) :
		episodeReward = 0
		currentState = worldEnv.reset()
		#print (currentState)
		for j in range(num_maxiter) :
			currentStateArray = stateDictionaryToArray(stateSpace, currentState)
			QValues = np.matmul(weightMatrix, currentStateArray) + bias
			action = np.argmax(QValues)

			isExplore = np.random.choice([0, 1], p = [1 - epsilon, epsilon])
			if isExplore == 1 :
				#print ("Random action")
				action = np.random.randint(3)
				
			nextState, reward, isDone = worldEnv.step(action) 
			episodeReward += reward

			newStateArray = stateDictionaryToArray(stateSpace, nextState)
			newQValues = np.matmul(weightMatrix, newStateArray) + bias
			newAction = np.max(newQValues)
			#print (isDone)
			tdTarget = reward + gamma*newAction
			tdDiff = QValues[action] - tdTarget
			tdDiff = alpha*tdDiff
			deltaWeightMatrix = np.zeros(weightMatrix.shape)
			deltaWeightMatrix[action,:] = currentStateArray
			#print (deltaWeightMatrix)
			weightMatrix = weightMatrix - tdDiff * deltaWeightMatrix
			bias = bias - tdDiff
			#print (bias)
			if isDone == True :
				break
			else :
				currentState = nextState
		rewardList = np.append(rewardList, episodeReward)
	saveRewardFile(rewardList, rewardFileName)
	saveWeightFile(bias, weightMatrix, weightFileName)

if __name__ == "__main__":
    main(sys.argv)