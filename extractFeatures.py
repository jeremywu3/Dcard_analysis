import json
import os

def readKeyWordFile():
	with open("sexhot.json","r") as fp:
		hotTop200 = json.load(fp)

	with open("sexnormal.json","r") as fp:
		norTop100 = json.load(fp)
	return hotTop200,norTop100



def combineKeywords(hotTop200,norTop100):
	keyWords = set()
	for hot in hotTop200:
		if '.' not in hot[0]:
			keyWords.add(hot[0])
	for nor in norTop100:
		if '.' not in nor[0]:
			keyWords.add(nor[0])
	keyWords.discard('')

	return keyWords

def createFeatureFile(keyWords,dirName):
	outputFileName = "{0}Data.csv".format(dirName)
	oFile = open(,"w")
	print("gender,hasSchool,reply,numImg,withNickname",file=oFile,end='')
	for key in keyWords:
		print(key)
		print(',{0}'.format(key),file=oFile,end='')
	print(',likeCount',file=oFile)
	for (dirPath, dirName, fileNames) in os.walk(dirN):
		for fileName in fileNames:
			fullFileName = '%s/%s' % (dirPath,fileName)
			with open(fullFileName) as fp:
				articles = json.load(fp)
			for article in articles:
				gender = 1 if article['gender'] == "M" else 0
				hasSchool = 1 if 'school' in article else 0
				reply = 1 if article['replyId'] else 0
				numImg = len(article['media'])
				excerptLen = len(article['excerpt'])
				withNickname = 1 if article['withNickname'] else 0
				print('{0},{1},{2},{3},{4}'.format(gender,hasSchool,reply,numImg,withNickname),file=oFile,end='')
				for key in keyWords:
					exist = 1 if key in article['tags'] else 0
					print(',{0}'.format(exist),file=oFile,end='')
				print(',{0}'.format(article['likeCount']),file=oFile)
	fp.close()

hotTop200,norTop100 = readKeyWordFile()
keyWords = combineKeywords(hotTop200,norTop100)
createFeatureFile(keyWords,"sex")
