import json

def extractBest(filename):
	with open(filename+".json",'r') as fp:
		obj = json.load(fp)

	result = dict()
	for model in obj:
		best_ = -1
		best_cm = None
		sum_acc = 0
		sum_1 = 0
		sum_2 = 0
		for cm in obj[model]:
			acc = (cm[0][0] + cm[1][1]) / float(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1])
			v1 = cm[1][1] / float(cm[0][1]+cm[1][1])
			v2 = cm[1][1] / float(cm[1][0]+cm[1][1])
			if best_ < v1:
				best_ = v1
				best_cm = cm
			sum_acc += acc
			sum_1 += v1
			sum_2 += v2
		result[model] = dict()
		result[model]['bestCM'] = best_cm
		result[model]['acc'] = sum_acc / len(obj[model])
		result[model]['v1'] = sum_1 / len(obj[model])
		result[model]['v2'] = sum_2 / len(obj[model])

	with open(filename+"Par.json","w") as fp:
		json.dump(result,fp)

def outputDatfile(filename):
	with open(filename+".json") as fp:
		obj = json.load(fp)
	fp = open(filename+'.dat','w')
	for model in obj:
		print('%s' % model,end=' ',file=fp)
		for key in ['acc','v1','v2']:
			print(',%f' % obj[model][key],end=' ',file=fp)
		print(file=fp)

extractBest("nobefore")
extractBest("noonce")
extractBest("notwice")

outputDatfile("nobeforePar")
outputDatfile("nooncePar")
outputDatfile("notwicePar")