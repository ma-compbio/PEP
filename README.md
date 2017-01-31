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
Functions include but are limited to:
--run_motif
  EPI prediction and evaluation using cross validation for PEP-Motif
--run_word
  EPI prediction and evaluation using cross validation for PEP-Word
--run_integrate
  EPI prediction and evaluation using cross validation for PEP-Integrate
--run_shuffle
  Shuffle feature orders for testing of PEP modules and repeating feature importance estimation to address that estimated importance can be affected by the order of features if they have equal predicitive effect
--parametered_cv
  Cross validation using gradient tree boosting

genVecs.py 
Generate feature vectors for enhancer-promoter pairs based on the word embedding model and weighted pooling in PEP-Word
Train a word embedding model in an unsupervised way given the samples generated using genUnlabelData.py

processSeq.py
Perform pre-processing of DNA sequences
The functions include:
-- extract DNA sequences from enhancer regions or promoter regions
-- extract word (K-mers) from enhancer or promoter DNA sequences

genLabelData.py.
Generate paired enhancer-promoter samples with labels from the EPI datasets to prepare for supervised training in PEP modules
Labels indicate whether a sample is positive (interacting enhancer/promoter pair) or negative (non-interaction enhancer/promoter pair)

genUnlabelData.py
Generate sentences (list of words/K-mers) from enhancer or promoter regions without labels for training a word embedding model in the enhancers or promoters respectively


************************************************************************************
# Required pre-installed packages
PEP requires the following packages to be installed:
- Python (tested on version 2.7.12)
- XGBoost (available from https://github.com/dmlc/xgboost)
- scikit-learn (tested on version 0.17)
- pandas (tested on version 0.17.1)


