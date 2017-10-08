import json
import os
import operator

def readRawfiles():
	dirN = 'data'
	GE1000 = 0
	Total = 0
	hotCounts = dict()
	norCounts = dict()
	keys = set()
	for (dirPath, dirName, fileNames) in os.walk(dirN):
		for fileName in fileNames:
			fullFileName = '%s/%s' % (dirPath,fileName)
			with open(fullFileName) as fp:
				articles = json.load(fp)
			for article in articles:
				Total += 1
				if article['likeCount'] >= 1000:
					GE1000 += 1
				if article['withNickname']:
					print(article['withNickname'])
				for key in article.keys():
					keys.add(key)
				tags = article['tags']
				for word in tags:
					if article['likeCount'] >= 1000:
						if word in hotCounts:
							hotCounts[word] += 1
						else:
							hotCounts[word] = 1
					if word in norCounts:
						norCounts[word] += 1
					else:
						norCounts[word] = 1
	print(Total)
	print(GE1000)
	print(keys)
	return norCounts,hotCounts

def countKeyWords(norCounts,hotCounts):	
	norTop100 = sorted(norCounts.items(), key=operator.itemgetter(1), reverse=True)[:100]
	hotTop200 = sorted(hotCounts.items(), key=operator.itemgetter(1), reverse=True)[:200]
	with open('normal.json','w') as fp:
		json.dump(norTop100,fp,ensure_ascii=False)
	with open('hot.json','w') as fp:
		json.dump(hotTop200,fp,ensure_ascii=False)

def main():
	norCounts,hotCounts = readRawFiles()
	countKeyWords(norCounts,hotCounts)