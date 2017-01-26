#-*- coding: utf-8 -*-  
import sys
import random
import numpy as np

def countCG(strs):
    strs = strs.upper()
    return float((strs.count("C")+strs.count("G")))/(len(strs))

# Read sequences as strings ("N" retained)
def getString(fileStr):
    file = open(fileStr, 'r')
    gen_seq = ""
    lines = file.readlines()
    for line in lines:
        line = line.strip()
        gen_seq += line

    gen_seq = gen_seq.upper()
    return gen_seq

# Read sequences of format fasta ("N" removed)
def getStringforUnlabel(fileStr):
    file = open(fileStr, 'r')
    gen_seq = ""
    lines = file.readlines()
    for line in lines:
        if(line[0] == ">"):
            continue
        else:
            line = line.strip()
            gen_seq += line

    gen_seq = gen_seq.upper()
    gen_seq = gen_seq.replace("N", "")

    return gen_seq

def get_reverse_str(str):
    str = str.upper()
    str_new=""
    for i in xrange(len(str)):
        if(str[i]=="T"):
            str_new+="A"
        elif(str[i]=="A"):
            str_new+="T"
        elif(str[i]=="G"):
            str_new+="C"
        elif(str[i]=="C"):
            str_new+="G"
        else:
            str_new+=str[i]
    return str_new

# Get sequence of 2K+1 centered at pos
def getSubSeq(str, pos, K):
    n = len(str)
    l = pos - K
    r = pos + K + 1
    if l > r or l < 0 or r > n - 1:
        return 0

    elif "N" in str[l:r]:
        return 0

    return str[l:r]

# Get sequence of 2K+1 centered at pos
def getSubSeq2(str, pos, K):
    n = len(str)
    l = max(0, pos - K)
    r = min(n - 1, pos + K + 1)
    if l > r:
        print l, pos, r
        print "left pointer is bigger than right one"
        return 0

    return str[l:pos]+" "+str[pos]+" "+str[pos+1:r]

# Convert DNA to sentences with overlapping window of size K
def DNA2Sentence(dna, K):

    sentence = ""
    length = len(dna)

    for i in xrange(length - K + 1):
        sentence += dna[i: i + K] + " "

    # remove spaces
    sentence = sentence[0 : len(sentence) - 1]
    return sentence

# Convert DNA to sentences with overlapping window of size K in reverse direction
def DNA2SentenceReverse(dna, K):

    sentence = ""
    length = len(dna)

    for i in xrange(length - K + 1):
        j = length - K - i
        sentence += dna[j: j + K] + " "

    # remove spaces
    sentence = sentence[0 : len(sentence) - 1]
    return sentence

# Convert DNA to sentences with non-overlapping window of size K
def DNA2SentenceJump(dna, K,step):
    sentence = ""
    length = len(dna)

    i=0
    while i <= length - K:
        sentence += dna[i: i + K] + " "
        i += step
    return sentence

# Convert DNA to sentences with non-overlapping window of size K in reverse direction
def DNA2SentenceJumpReverse(dna, K,step):
    sentence = ""
    length = len(dna)

    i=0
    j=0
    while j <= length - K:
        i = length - K - j
        sentence += dna[i: i + K] + " "
        j += step
    return sentence
