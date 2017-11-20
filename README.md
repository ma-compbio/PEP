**********************************************************************************
# PEP (Prediction Enhancer Promoter interactions)

PEP is a framework for predicting long-range enhancer-promoter interactions (EPI) incorporating two strategies for extracting features directly from the DNA sequences of enhancer and promoter elements. There are three modules in PEP, which are PEP-Motif, PEP-Word and PEP-Integrate. PEP-Integrate combines selected features generated from PEP-Motif and PEP-Word. Gradient Tree Boosting is used in each of the three modules to training a predictor for EPIs based on respective feature representations of enhancer-promoter pairs. In PEP-Motif, we search for patterns of known transcription factor binding site (TFBS) motifs in the sequences involved in EPI. The normalized occurrence frequencies of these TFBS motifs are then used as features representing an enhancer or a promoter. In PEP-Word, we use the word embedding model to embed the sequences of enhancer and promoter regions into a new feature space. Each sequence is then represented by a continuous feature vector. In both PEP-Motif and PEP-Word modules, individual feature vectors of paired regions are concatenated to form feature representations of the given enhancer-promoter pair.  

PEP moduels are mainly trained and evaluated on the the E/P (Enhancer/Promoter) datasets used by TargetFinder[1]. The datasets consist of EPI and non-EPI samples in six cell lines (GM12878, K562, IMR90, HeLa-S3, HUVEC, and NHEK). PEP can also be applied to other EPI datasets where training data with known interactions between enhancers and promoters are available. 

***********************************************************************************
# File & Function Description  
PEP.py 
mainEntrance function and options of model setting parameters provided  

mainPEP.py 
Perform model training, EPI prediction and performance evaluation in PEP-Motif, PEP-Word, and PEP-Integrate
Functions included but not limited to:
- run_motif

  EPI prediction and evaluation using cross validation for PEP-Motif
  
- run_word

  EPI prediction and evaluation using cross validation for PEP-Word
  
- run_integrate

  EPI prediction and evaluation using cross validation for PEP-Integrate
  
- run_shuffle

  Shuffle feature orders for testing of PEP modules and repeating feature importance estimation to address the problem that estimated importance can be affected by the order of features if they have equal predicitive effect
  
- parametered_cv

  Cross validation using gradient tree boosting

genVecs.py 

Generate feature vectors for enhancer-promoter pairs based on the word embedding model and weighted pooling in PEP-Word
Train a word embedding model in an unsupervised way given the samples generated using genUnlabelData.py

processSeq.py

Perform pre-processing of DNA sequences
Functions included:

- extract DNA sequences from enhancer regions or promoter regions

- extract word (K-mers) from enhancer or promoter DNA sequences

genLabelData.py

Generate paired enhancer-promoter samples with labels from the EPI datasets to prepare for supervised training in PEP modules
Labels indicate whether a sample is positive (interacting enhancer/promoter pair) or negative (non-interaction enhancer/promoter pair)

genUnlabelData.py

Generate sentences (list of words/K-mers) from enhancer or promoter regions without labels for training a word embedding model for the enhancers or promoters respectively

************************************************************************************
# Usage
The command to use PEP for predicting enhancer promoter interactions is as follows:

python PEP.py [Options] 

- -f, --feature : the number of features of Word2Vec model, default = 300
- -g, --generate : to generate the data for word embedding model training or to use the ones before, default = true
- -t, --type : to use E/P (Enhancer/Promoter) data or EE/P (Extended Enhancer/Promoter) data, default = ep
- -c, --cell : the cell line used for training and evaluation, default = GM12878
- -k, --k: minimum occurences of a word in a sequence if the word can be used for word embedding model training, default = 1
- -w, --word : the length of word (K-mer) used for word embedding model training, default = 6
- -i, --integrate : to use integrated features or not, default = "false"
- -s, --sel : the number of motif features to be used in the feature integration (PEP-Integrate) mode, default=50
- -e, --thresh_mode: the mode of estimating threshold for the predictor: 0- default threshold (threshold = 0.5); 1- simple mode (randomly split the training data into 90% for training and 10% for validation to estimate a threshold); 2- estimate threshold by 5-fold inner round cross validation; default=1

Example: python PEP.py -c 'GM12878' -t 'ep' (using PEP-Word for training a word embedding model and performing enhancer promoter interaction prediction in cell line K562)

To use PEP-Word, please create two folders named "Data" and "DataVecs" in the same directory with the PEP source code.  
- Please create a subfolder named "Learning" in the folder "Data", and please place genome sequence data needed in "Learning". 
- Please create a subfolder for each cell line in the folder "Data" to place the annotations for enhancers/promoters and samples of enhancer-promoter pairs. For example, create a subfolder named "K562" and place the enhancers.bed, promoters.bed and training samples with given enhancer/promoter regions in the folder. 

We created the folders needed as an example. We provided some example data in the subfolder "GM12878" under "Data". The format of example data can be followed if you use your own data. The example data are from [1].
- The enhancers.bed and promoters.bed provide genome locations of all the annotated active enhancers and promoters in the corresponding cell line, which are used for training the word embedding model. There are four columns representing chromosome, start position, ending position, and the name of an enhancer/promoter respectively in the enhancers.bed or promoters.bed. Please add the column names "chromosome", "start", "end", and "name" if you use your own annotation files. 
- The file pairs_ep.csv comprises the positive and negative samples in cell line GM12878. 

If you use the command python PEP.py -c CellName -t 'ep', the procedures of PEP-Word will be performed, which involve unsupervised training of word embeding model, supervised training of a Gradient Tree Boosting (GTB) classifier and making predicitons of enhancer promoter interactions (EPIs) for the cell line specified by CellName. 
- Files named "unlabeled_trainraw_enhancer_CellName", "unlabeled_trainraw_promoter_CellName" and "supervised_CellName_ep" will be generated in folder Data/Learning. The first two files are used for training the word embedding model, and the third file is used for training the GTB based predictor. 
- "datavecs_CellName_ep.npy" will be generated in folder Datavecs, which comprises the feature representation of enhancer/promoter sequences obtained from the word embbeding models (generated in the same directory with source code, named as "CellName_enahcer" and "CellName_promoter") and TF-IDF dictionaries (generated in Data, named as "enhancertfidfCellName" and "enhancertfidfCellName") using weighted pooling.
- The prediction results will be output in the same directory of the source code.

If you want to begin a new experiment for the data of a cell line which you have used before, please remove all the old data in the folders Data/Learning and DataVecs of this cell line or place them in another directory. PEP will not overwrite the exsiting files in these two filefolders and will reuse the exsting files by default.

Please update the "genome_path" on line 37 of genLabelData.py to be the directory where the DNA sequence data of each chromosome of the studied genome are placed. PEP-Word will use the DNA sequence data in this directory to extract sequences for the enhancer and promoter regions.

Please change the cell name used in "unlabeled_train_enhancer_CellName" and "unlabeled_train_promoter_CellName" on line 21 and line 25 of genVecs.py to the CellName you use in the command line accordingly. 

To use PEP-motif, please prepare the motif frequency features of the enhancer and promoter pairs in the interested cell line. Please update the directory and filename of the motif feature data on line 303 of MainPEP.py. In our experiment we store the features in a Matlab MAT file named pairs_[Type (ep or eep)][CellName]_motif.mat. There are two fields in the MAT file: "seq_m" stores the feature vectors of the samples, and "lab_m" stores the corresponding class labels of the samples. The motif frequency features can be obtained by using motif scanning algorithms to search for possible motifs in the enhancer or promoter regions with a motif database and computing motif frequencies normalized by the enhancer/promoter length. Please refer to the paper "Exploiting sequence-based features for predicting enhancer–promoter interactions" for detailed descriptions. 

The motif frequency feature data we prepared and used for PEP-Motif are available at http://genome.compbio.cs.cmu.edu/~yy3/PEP/motif_feature/. 
There is a file folder for each studied cell line. There are three files in each file folder of a cell line:
- pairs_ep[CellName]_motif.mat : feature matrix of the size sample_num * feature_dimension. Each column corresponds to a motif. The feature vector of each sample is the concatenation of the feature in the enhancer region (with 4kb flanking regions) and the feature in the promoter region. The feature dimension varies between different cell lines. 
- pairs_ep[CellName]_motif_serial.mat : The serial number of each sample in the feature matrix of pairs_ep[CellName]_motif.mat, which is in the same order as the rows of the feature matrix. The serial numbers are used to match the features generated in PEP-Word and the features generated in PEP-Motif for PEP-Integrate, if PEP-Integrate is called. PEP-Word and PEP-Motif generated features for samples in different orders, so we prepared the serial numbers to match the features of the same sample. 
- ep[CellName]_sel_union_balanced_inter.mat: The ranking of motif features based on their feature importance estimated by PEP-Motif. The ranking is in the ascending order. Each number represents a column in the feature matrix (index starts from 0). For example, the first number represents the most important motif feature estimated by PEP-Motif. The number k represents the motif corresponding to the (k+1)th column of the feature matrix. This feature importance ranking data are used to select a subset of PEP-Motif features to combine with PEP-Word features in PEP-Integrate, if PEP-Intergrate is called. 

To use the motif feature data and the auxiliary data, please place them in the same directory with the source code. If you're using your own motif features, please change the paths and filenames of the motif features in PEP.py accordingly. 

References:

[1] S. Whalen, R. M. Truty, and K. S. Pollard. Enhancer-promoter interactions are encoded by complex genomic signatures on looping chromatin. Nature genetics, 48(5):488–496, 2016.

************************************************************************************
# Required pre-installed packages
PEP requires the following packages to be installed:
- Python (tested on version 2.7.12)
- XGBoost (available from https://github.com/dmlc/xgboost)
- scikit-learn (tested on version 0.17)
- pandas (tested on version 0.17.1)
- numpy (tested on version 1.11.1)
- gensim (tested on version 0.13.1)

You could install the Anaconda (avilable from https://www.continuum.io/downloads) for convenience, which provides a open souce collection of widely used data science packages including Python and numpy. PEP is tested using Anaconda 4.1.1.

