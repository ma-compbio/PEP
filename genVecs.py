#encoding:utf-8
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import pandas as pd
import numpy as np
import os
import sys
import math
import random

import processSeq
import warnings
import threading
from multiprocessing.dummy import Pool as ThreadPool
from sklearn import preprocessing
import sklearn.preprocessing
from gensim import corpora, models, similarities

class mycorpuse(object):
	def __iter__(self):
		for line in open("./Data/Learning/unlabeled_train_enhancer_GM12878"):
			yield line.split()
class mycorpusp(object):
	def __iter__(self):
		for line in open("./Data/Learning/unlabeled_train_promoter_GM12878"):
			yield line.split()

# Load training data
def getData(type,cell):
	data = pd.read_table('./Data/Learning/supervised_'+str(cell)+"_"+str(type))
	return data

# Load trained Word2Vec model or train a new model
def getWord_model(word,num_features,min_count,type,cell):
	word_model1 = ""
	model_name = str(cell)+"_enhancer"
	#model_name = "pe_6"
	
	if not os.path.isfile("./" + model_name):
		sentence = LineSentence("./Data/Learning/unlabeled_train_enhancer_"+str(cell),max_sentence_length=15000)
		print "Start Training Word2Vec model..."
		# Set values for various parameters
		num_features = int(num_features)	  # Word vector dimensionality
		min_word_count = int(min_count)	  # Minimum word count
		num_workers = 20		 # Number of threads to run in parallel
		context = 20			# Context window size
		downsampling = 1e-3	 # Downsample setting for frequent words

		# Initialize and train the model
		print "Training Word2Vec model..."
		word_model1 = Word2Vec(sentence, workers=num_workers,\
						size=num_features, min_count=min_word_count, \
						window =context, sample=downsampling, seed=1)
		word_model1.init_sims(replace=False)
		word_model1.save(model_name)
		print word_model1.most_similar("CATAGT")
	else:
		print "Loading Word2Vec model..."
		word_model1 = Word2Vec.load(model_name)

	word_model2 = ""
	model_name = str(cell)+"_promoter"
	if not os.path.isfile("./" + model_name):
		sentence = LineSentence("./Data/Learning/unlabeled_train_promoter_"+str(cell),max_sentence_length=15000)
		
		print "Start Training Word2Vec model..."
		# Set values for various parameters
		num_features = int(num_features)	  # Word vector dimensionality
		min_word_count = int(min_count)	  # Minimum word count
		num_workers = 20		 # Number of threads to run in parallel
		context = 20			# Context window size
		downsampling = 1e-3	 # Downsample setting for frequent words

		# Initialize and train the model
		print "Training Word2Vec model..."
		word_model2 = Word2Vec(sentence, workers=num_workers,\
						size=num_features, min_count=min_word_count, \
						window=context, sample=downsampling, seed=1)
		word_model2.init_sims(replace=False)
		word_model2.save(model_name)
		print word_model2.most_similar("CATAGT")
	else:
		print "Loading Word2Vec model..."
		word_model2 = Word2Vec.load(model_name)

	return word_model1,word_model2

# Split sequences into words
def getCleanDNA_split(DNAdata,word):

	dnalist = []
	counter = 0
	for dna in DNAdata:
		if counter % 100 == 0:
			print "DNA %d of %d\r" % (counter, len(DNAdata)),
			sys.stdout.flush()

		dna = str(dna).upper()
		dnalist.append(processSeq.DNA2Sentence(dna,word).split(" "))

		counter += 1
	print
	return dnalist

def makeFeatureVecs(words, model, num_features,word,k,temp):
	featureVec = np.zeros((k,num_features), dtype="float32")
	nwords = 0
	index2word_set = set(model.index2word)
	length = len(words)
	for word in words:
		if word in index2word_set:
		# divide the words into k parts, add up in each part
			featureVec[math.floor((nwords * k) / length)] += (model[word]) * temp[nwords]
			nwords =nwords + 1

	featureVec = featureVec.reshape(k * num_features)
	#featureVec = featureVec/nwords
	return featureVec

def mean2max(vec):
	length = len(vec)
	mean1 = np.max(vec[0:int(length*0.5)],axis = 0)
	mean2 = np.max(vec[int(length*0.5):int(length)],axis = 0)
	maxvec = np.mean([mean1,mean2],axis = 0)
	return maxvec

def getAvgFeatureVecs(data,model1,model2, num_features, word,k,type,cell):
	dnaFeatureVecs = np.zeros((len(data),2*k*num_features), dtype="float32")
	if not os.path.isfile("./Data/enhancertfidf"+str(cell)):
		print "Getting dictionary"
		Corp = mycorpuse()
		dictionary = corpora.Dictionary(Corp)
		dictionary.save("./Data/enhancerdic"+str(cell))
		corpus = [dictionary.doc2bow(text) for text in Corp]
		print "Calculating TFIDF"
		tfidf = models.TfidfModel(corpus)
		tfidf.save("./Data/enhancertfidf"+str(cell))
	else:
		tfidf = models.TfidfModel.load("./Data/enhancertfidf"+str(cell))
		dictionary = corpora.Dictionary.load("./Data/enhancerdic"+str(cell))
	dict1 = {k:v for k, v in dictionary.items()}

	DNAdata1 = getCleanDNA_split(data["seq1"],word)

	counter = 0
	for dna in DNAdata1:

		if counter % 100 == 0:
			print "DNA %d of %d\r" % (counter, len(DNAdata1)),
			sys.stdout.flush()
		
		vec_bow = dictionary.doc2bow(dna)
		vec_tfidf = tfidf[vec_bow]
		
		for i in xrange(len(vec_tfidf)):
			dnaFeatureVecs[counter][0:k*num_features] += model1[dict1[vec_tfidf[i][0]]] * vec_tfidf[i][1]
		
		#dnaFeatureVecs[counter][0:k*num_features] = makeFeatureVecs(DNA, model1, num_features,word,k,temp)
		#dnaFeatureVecs[counter][0:k*num_features] = np.mean(model1[DNA],axis = 0)
		#dnaFeatureVecs[counter][0:k*num_features] = mean2max(model1[DNA])
		#dnaFeatureVecs[counter][0:k*num_features] = np.max(model1[DNA],axis = 0)
		counter += 1
	
	print
	del DNAdata1

	counter = 0
	if not os.path.isfile("./Data/promotertfidf"+str(cell)):
		print "Getting dictionary"
		Corp = mycorpusp()
		dictionary = corpora.Dictionary(Corp)
		dictionary.save("./Data/promoterdic"+str(cell))
		corpus = [dictionary.doc2bow(text) for text in Corp]
		print "Calculating TFIDF"
		tfidf = models.TfidfModel(corpus)
		tfidf.save("./Data/promotertfidf"+str(cell))
	else:
		tfidf = models.TfidfModel.load("./Data/promotertfidf"+str(cell))
		dictionary = corpora.Dictionary.load("./Data/promoterdic"+str(cell))
	
	dict2 = {k:v for k, v in dictionary.items()}

	DNAdata2 = []
	counter = 0
	for dna in data["seq2"]:
		if counter % 100 == 0:
			print "DNA %d of %d\r" % (counter, len(data)),
			sys.stdout.flush()

		dna = str(dna).upper()
		DNAdata2.append(processSeq.DNA2Sentence(dna,word).split(" "))

		counter += 1

	counter = 0
	print

	for dna in DNAdata2:
		if counter % 100 == 0:
			print "DNA %d of %d\r" % (counter, len(DNAdata2)),
			sys.stdout.flush()

		vec_bow = dictionary.doc2bow(dna)
		vec_tfidf = tfidf[vec_bow]
		
		for i in xrange(len(vec_tfidf)):
			dnaFeatureVecs[counter][k*num_features:2*k*num_features] += model2[dict2[vec_tfidf[i][0]]] * vec_tfidf[i][1]
		
		#dnaFeatureVecs[counter][k*num_features:2*k*num_features] = makeFeatureVecs(dna, model2, num_features,word,k,temp)
		#dnaFeatureVecs[counter][k*num_features:2*k*num_features] = np.mean(model2[dna],axis = 0)
		#dnaFeatureVecs[counter][k*num_features:2*k*num_features] = mean2max(model2[dna])
		#dnaFeatureVecs[counter][k*num_features:2*k*num_features] = np.max(model1[dna],axis = 0)
		counter += 1
	print
	np.save("./Datavecs/datavecs_"+str(cell)+"_"+str(type)+".npy",dnaFeatureVecs)
	return dnaFeatureVecs

def run(word, num_features,K,type,cell):
	warnings.filterwarnings("ignore")

	global word_model,data,k

	word = int(word)
	num_features = int(num_features)
	k=int(K)
	word_model=""
	min_count=10

	word_model1,word_model2 = getWord_model(word,num_features,min_count,type,cell)

	# Read data
	data = getData(type,cell)

	length = data.shape[0]
	print length

	print "Generating Training and Testing Vector"
	dataDataVecs = getAvgFeatureVecs(data,word_model1,word_model2,num_features,word,k,type,cell)


if __name__ == "__main__":
	run(6,300,1,'new','GM12878')
