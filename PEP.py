#-*- coding: utf-8 -*-  
from optparse import OptionParser
import genLabelData,genUnlabelData,mainPEP,genVecs
import os.path

def parse_args():
	parser = OptionParser(usage="Enhancer Promoter Interaction Prediction", add_help_option=False)
	parser.add_option("-f", "--feature", default="300", help="Set the number of features of Word2Vec model")
	parser.add_option("-g","--generate",default="true", help="Generate the Data or Use the ones before")
	parser.add_option("-t","--type",default="ep",help="eep data or ep data")
	parser.add_option("-c","--cell",default = "GM12878",help="Cell Line")
	parser.add_option("-k","--k",default="1",help="k")
	parser.add_option("-w","--word",default = "6")
	parser.add_option("-i","--integrate",default="false", help="Use integrated features or not")
	parser.add_option("-s","--sel",default=50, help="The number of motif feature to be used in the combination mode")
	parser.add_option("-e","--thresh_mode",default=1,help="The mode of estimating threshold:0-default;1-simple mode;2-cv mode")

	(opts, args) = parser.parse_args()
	return opts


def run(word,num_features,generate,type,cell,k,integrate,sel,thresh_mode):

	if(os.path.exists("./Data/Learning")==False):
		os.makedirs("./Data/Learning")

	print "parameters are as followed\n" \
		  "feature=%r\tgenerate=%r\n" \
		  "type=%r\tcell=%r\n" \
		  "k=%r\n"\
		  %(num_features,generate,type,cell,k)

	if generate == "true":
		if not os.path.isfile("./Data/Learning/supervised_"+str(cell)+"_"+str(type)):
			genLabelData.run(type,cell)
		if not os.path.isfile("./Data/Learning/unlabeled_train_promoter_"+str(cell)+"_"+str(type)):
			genUnlabelData.run(type,cell,word)
		if not os.path.isfile("./Datavecs/datavecs_"+str(cell)+"_"+str(type)+".npy"):
			genVecs.run(word,num_features,k,type,cell)
	
	word = int(word)
	num_features = int(num_features)
	k=int(k)
	sel=int(sel)
	thresh_mode=int(thresh_mode)
	
	if integrate == "false":
		mainPEP.run_word(word,num_features,k,type,cell,thresh_mode)
		# mainPEP.run_motif(type,cell,thresh_mode)
	else:
		mainPEP.run_integrate(word,num_features,k,type,cell,sel,thresh_mode)


def main():
	opts = parse_args()
	run(opts.word,opts.feature,opts.generate,opts.type,opts.cell,opts.k,opts.integrate,opts.sel,opts.thresh_mode)

if __name__ == '__main__':
	main()
