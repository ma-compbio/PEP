#-*- coding: utf-8 -*-  
import processSeq
import os
import pandas as pd

# Generate unlabeled data
def run(type,cell,word):
    word = int(word)
    if not os.path.isfile("./Data/Learning/unlabeled_train_enhancer_"+str(cell)):
        print "Generating unlabeled data..."
        outfile = open("./Data/Learning/unlabeled_train_enhancer_"+str(cell), "w")
        file = open("./Data/Learning/unlabeled_trainraw_enhancer_"+str(cell), 'r')
        gen_seq = ""
        lines = file.readlines()
        for line in lines:
            line = line.strip()
            sentence = processSeq.DNA2Sentence(line,word)
            outfile.write(sentence+"\n")
        
        outfile = open("./Data/Learning/unlabeled_train_promoter_"+str(cell), "w")
        file = open("./Data/Learning/unlabeled_trainraw_promoter_"+str(cell), 'r')
        gen_seq = ""
        lines = file.readlines()
        for line in lines:
            line = line.strip()
            sentence = processSeq.DNA2Sentence(line,word)
            outfile.write(sentence+"\n")

def main():
    print "main"

if __name__ == '__main__':
    run()
