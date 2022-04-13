#To convert from original Ontonotes data given by LDC to BIO format.
#It also generates source based subsets of train/dev/test sets. 
#slightly modified version from: https://github.com/yuchenlin/OntoNotes-5.0-NER-BIO

import glob,os,itertools


def write_bio(fnames, genre, group):
    text =  ""
    for cur_file in fnames: 
        with open(cur_file, 'r') as f:
            #print(cur_file)
            flag = None
            for line in f.readlines():
                l = line.strip()
                l = ' '.join(l.split())
                ls = l.split(" ")
                if len(ls) >= 11:
                    word = ls[3]
                    pos = ls[4]
                    cons = ls[5]
                    ori_ner = ls[10]
                    ner = ori_ner
                    # print(word, pos, cons, ner)
                    if ori_ner == "*":
                        if flag==None:
                            ner = "O"
                        else:
                            ner = "I-" + flag
                    elif ori_ner == "*)":
                        ner = "I-" + flag
                        flag = None
                    elif ori_ner.startswith("(") and ori_ner.endswith("*") and len(ori_ner)>2:
                        flag = ori_ner[1:-1]
                        ner = "B-" + flag
                    elif ori_ner.startswith("(") and ori_ner.endswith(")") and len(ori_ner)>2 and flag == None:
                        ner = "B-" + ori_ner[1:-1]

                    text += "\t".join([word, pos, cons, ner]) + '\n'
                else:
                    text += '\n'
            text += '\n'
            # break

    outputpath = os.path.join("bio/"+group,"onto."+genre+".ner")
    with open(outputpath, 'w') as f:
        f.write(text)

maindir = "data/"
#Note: This folder already has only English data in the form data/train or test or development/
#followed by the ontonotes structure. (i.e., genre wise subfolders)

subdirs = ["train", "test", "development"]

genres = ["bc", "bn", "mz", "nw", "pt", "tc", "wb"]

for subdir in subdirs:
   print("stats for ", subdir)
   for genre in genres:
      fnames = []
      for root, dirs, files in os.walk(os.path.join(maindir, subdir, genre)):
         for f in files:
            fnames.append(os.path.join(root, f))
	    #print("file names for this genre: ")
	    #print(fnames)
            write_bio(fnames, genre, subdir)
         print(root)
      print("number of files for", genre, "genre: ", str(len(fnames)))

