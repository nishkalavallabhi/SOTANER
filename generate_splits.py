#Generate multiple random train/dev/test splits for Ontonotes dataset.
from sklearn.model_selection import KFold, train_test_split, ShuffleSplit
import random
import pprint

#CHANGE THESE TWO LINES BEFORE RUNNING, TO APPROPRIATE PATHS
outputfolder =  "bio-splits/"
inputfilepath = "onto.everything.ner" #combined onto notes train, test, and dev into one file.

myds = {}

"""
store as: {sent id: string} (where string is the string read from the files - bio formatted 4 cols)
"""
def make_fileids():
    myds = {}
    sentid = 0
    fh = open(inputfilepath, encoding="utf-8")
    fh.readline()
    temp = ""
    for line in fh:
        if line.strip() is not "": #if line is not empty, which indicates sentence is continuing
            temp += line.strip()+"\n"
        else: #potentially a new sentence will start here.
            if temp is not "":
                myds[sentid] = temp.strip()
                temp = ""
                sentid +=1
            #else, don't do anything, just pass - this is a spurious empty line.
    fh.close()
    return myds

"""
Just using this to see if the making dict function works. 
"""
def test_makefileids():
    temp = make_fileids()
    print(len(temp))
    for i  in range(5):
        sentid, taggedsen = random.choice(list(temp.items()))
        print(taggedsen)
        print()

"""
Generate the train/dev/test file for a given fold
"""
def generate_file(filepath, idslist):
    fh = open(filepath, "w", encoding="utf-8")
    for id in idslist:
        fh.write(myds[id])
        fh.write("\n\n")
    fh.close()

"""
creates train/dev/test files for a given fold. calls generate_file which actually generates.
train, dev, test are lists with sentids from the dict created by make_fileids
"""
def make_filenames(train, dev, test, foldid):
    train_file = outputfolder+"fold"+str(foldid)+"_train.ner"
    dev_file = outputfolder + "fold" + str(foldid) + "_dev.ner"
    test_file = outputfolder + "fold" + str(foldid) + "_test.ner"
    generate_file(train_file, train)
    generate_file(dev_file, dev)
    generate_file(test_file, test)

"""
main function to call once the dictionary is created.
This makes splits, and calls functions to generate files per split.
"""
def make_splits(dictlen):
    templs = list(range(0, dictlen))
    tempfolds = KFold(n_splits=10, shuffle=True)
    shushi = ShuffleSplit(n_splits=1, test_size=0.1)
    fold = 1
    for temptrain,test in tempfolds.split(templs):
        train,dev = next(shushi.split(temptrain))
        print(len(train), len(dev), len(test), fold)
        make_filenames(train, dev, test, fold)
        fold +=1

#test_makefileids()
myds = make_fileids()
dictlen = len(myds)
make_splits(dictlen)

""""""
