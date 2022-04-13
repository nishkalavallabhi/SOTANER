#!/usr/bin/env python
# coding: utf-8

# This code contains functions to use a pre-trained SparkNLP model (nerdl, with bert_base in this case) and  getting its performance for a given test set. 
# 
# This proceeds in two steps:
# a) convert the test set to SparkNLP's format
# b) use the nerdl model and make predictions.
# 
# Make changes to the filepaths appropriately in the inputdir/outputdir/testfiles variables and you should be all set!

from pyspark.sql import SparkSession
from pyspark.ml import Pipeline

import sparknlp
from sparknlp.annotator import *
from sparknlp.common import *
from sparknlp.base import *
from sparknlp.training import CoNLL

import pandas as pd
import os

# to use GPU 
spark = sparknlp.start()

print("Spark NLP version: ", sparknlp.version())
print("Apache Spark version: ", spark.version)


#CHANGE THESE APPROPRIATELY EACH TIME YOU WANT TO DO BLACKBOX TESTING OF A TEST SET
inputdir = 'conll-2012-share/bio/test/'
outputdir = 'spark-format-test'
testfiles = ['onto.bn.ner', 'onto.bc.ner', 'onto.mz.ner', 'onto.nw.ner', 'onto.tc.ner', 'onto.wb.ner']

#Set up the NER model
bert = BertEmbeddings.pretrained('bert_base_cased', 'en').setInputCols(["sentence",'token']).setOutputCol("bert").setCaseSensitive(True).setMaxSentenceLength(512)
#Set up the Pre-trained NER model
ner_onto = NerDLModel.pretrained('onto_bert_base_cased', lang='en') \
        .setInputCols(["sentence", "token", "bert"])\
        .setOutputCol("ner")

nlp_pipeline = Pipeline(stages=[bert,ner_onto])


#Converts our regular CONLL files to SparkNLP's CONLL format
def convert_format(inputpath, outputpath):
    # create the training file
    with open(inputpath) as fp:
        text = fp.readlines()
    text = "".join(text[1:]).split("\n\n") 
    df = pd.DataFrame([x.split('\t') for x in text[1].split('\n')], 
                      columns=["Token","Pos","Pos_special","Entity_label"])
    
    # creating the training data
    conll_lines = "-DOCSTART- -X- -X- -O-\n\n"
    for t in range(len(text)):    
        df = pd.DataFrame([x.split('\t') for x in text[t].split('\n') if len(x.split('\t')) == 4], columns=["Token","Pos","Pos_special","Entity_label"])
        tokens = df.Token.tolist()
        pos_labels = df.Pos.tolist()
        entity_labels = df.Entity_label.tolist()
        for token, pos, label in zip(tokens,pos_labels,entity_labels):
            conll_lines += "{} {} {} {}\n".format(token, pos, pos, label)
        conll_lines += "\n"
        
    with open(outputpath,"w") as fp:
        for line in conll_lines:
            fp.write(line)
    
    print("Done")

"""
Runs the NER model over a testfile and makes predictions. 
"""
def get_results(testfile):
    test_data = CoNLL().readDataset(spark, testfile)
    myNerModel = nlp_pipeline.fit(test_data)
    results = myNerModel.transform(test_data).select("sentence","token","label","ner").collect()
    
    #test_data.show()
    
    # to find exceptions where no. of labels does not match no. of ners detected
    count = 0
    indices = []
    for i,row in enumerate(results):
        if len(row['label']) != len(row['ner']):
            count += 1
            indices.append(i)

    print(count)
    print(indices)

    exclusion_list = [results[t] for t in indices]
    results = [results[i] for i in range(len(results)) if i not in indices]
    
    tokens = []
    labels = []
    ners = []

    for row in results:
        tokens.append([t['result'] for t in row['token']])
        labels.append([t['result'] for t in row['label']])
        ners.append([t['result'] for t in row['ner']])

    from seqeval.metrics import accuracy_score, f1_score, classification_report
    #print(accuracy_score(labels,ners))
    #print(f1_score(labels,ners))

    print(classification_report(labels,ners, zero_division=1,digits=6))

    
#Convert all test files into SparkNLP's expected CONLL format.
for testfile in testfiles:
    convert_format(os.path.join(base_path,testfile), os.path.join(outputdir,testfile))#

for testfile in testfiles:
    filepath = os.path.join(outputdir,testfile)
    print(filepath)
    get_results(filepath)
    print("**************")
