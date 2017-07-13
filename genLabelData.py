#-*- coding: utf-8 -*-  
import pandas as pd
import processSeq
import random
import sys
import os
import numpy as np

def run(type,cell):
	print "Generating labeled data..."
	table_name = "./Data/"+str(cell)+"/pairs_"+str(type)+".csv"
	table = pd.read_table(table_name, sep=',')

	label_file = open("./Data/Learning/supervised_"+str(cell)+"_"+str(type), "w")
	unlabel_enhancer = open("./Data/Learning/unlabeled_trainraw_enhancer_"+str(cell), "w")
	unlabel_promoter = open("./Data/Learning/unlabeled_trainraw_promoter_"+str(cell), "w")


	label_file.write("label\t"
				  "seq1\t"
				  "seq2\t"
				  "chr\t"
				  "num\n")

	total = len(table)

	list = ["chr1", "chr2", "chr3", "chr4", "chr5", "chr6", "chr7", "chr8", "chr9", \
			"chr10", "chr11", "chr12", "chr13", "chr14", "chr15", "chr16", "chr17", \
			"chr18", "chr19", "chr20", "chr21", "chr22", "chrX", "chrY","chrM"]

	print 'Start making data'

	number_positive = 0
	dict_pos={}


	genome_path = "./"    # Please add the directory where the genome sequence data are placed
	# Generate positive samples
	for i in xrange(total):
		
		if (number_positive % 1000 == 0) and (number_positive != 0):
			print "number of positive: %d of %d\r" %(number_positive,total),
			sys.stdout.flush()

		chromosome = table["enhancer_chrom"][i]

		# Check if there is the chromosome in the dictionary. If not input sequence of the chromosome
		if dict_pos.has_key(chromosome):
			strs = dict_pos[chromosome]
		else:
			strs = processSeq.getString(genome_path + chromosome + ".fa")
			dict_pos[chromosome] = strs

		enhancer_start = table["enhancer_start"][i] - 1
		enhancer_end = table["enhancer_end"][i]

		promoter_start = table["promoter_start"][i] - 1
		promoter_end = table["promoter_end"][i]

		if str(type) == "ep":
			enhancer_start -= 4000
			enhancer_end += 4000

		edstrs1 = strs[enhancer_start:enhancer_end]
		edstrs2 = strs[promoter_start:promoter_end]

		if "N" in edstrs1 or "N" in edstrs2:
			table = table.drop(i)
			continue

		outstr = "%s\t%s\t%s\t%s\t%s\n"%(table["label"][i],edstrs1,edstrs2,str(chromosome),table["active_promoters_in_window"][i])
		label_file.write(outstr)
		number_positive += 1
	print
	table.to_csv(table_name,index = False)
	
	table = pd.read_table("./Data/"+str(cell)+"/enhancers.bed")
	for i in xrange(len(table)):
		chromosome = table["chromosome"][i]
		start = table["start"][i] - 1
		end = table["end"][i]

		if str(type) == "eep":
			start += 3000
			end -= 3000
		strs = dict_pos[chromosome]
		edstrs = strs[start:end] + "\n"
		unlabel_enhancer.write(edstrs)
	
	table = pd.read_table("./Data/"+str(cell)+"/promoters.bed")
	for i in xrange(len(table)):
		chromosome = table["chromosome"][i]
		start = table["start"][i] - 1
		end = table["end"][i]
		strs = dict_pos[chromosome]
		edstrs = strs[start:end] + "\n"
		unlabel_promoter.write(edstrs)

if __name__ == "__main__":
	run()
