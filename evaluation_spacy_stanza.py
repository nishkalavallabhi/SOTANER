#Evaluating performance with Spacy/Stanza (outside command line tools)
import spacy, stanza, spacy_stanza
from spacy.tokenizer import Tokenizer
from seqeval.metrics import classification_report
from seqeval.metrics import f1_score
import sys
import itertools


"""
Reads an ontonotes test file, and stores it as two lists of lists: 
one each for sentences (as tokens) and their NER tags.
TODO: Should think about a better data structure here.
This works with 4 col BIO format. If the data is 3 col, change the first line.
"""
def read_file(filepath):
    numcols = 4 #change here for 3 col vs 4 col conll format.
    fh = open(filepath)
    sentences = []
    netags = []
    tempsen = []
    tempnet = []
    for line in fh:
       if line.strip() == "":
          if tempsen and tempnet:
              sentences.append(tempsen)
              netags.append(tempnet)
              tempsen = []
              tempnet = []
       else:
          splits = line.strip().split("\t")
          tempsen.append(splits[0])
          tempnet.append(splits[numcols-1])
    fh.close()
    print("Num sentences in: ", filepath, ":", len(sentences))
    return sentences, netags

"""
Takes a BIO formatted test set path, and evaluates Spacy and Stanza's existing NER models.
For spacy, change the model if needed. I tried with _trf, and _lg.
"""
def spacy_and_stanza(mypath):

    #setting up spacy and stanza
    nlp = spacy.load("en_core_web_trf") #can try with other models later
    nlp.tokenizer = Tokenizer(nlp.vocab)
    snlp = spacy_stanza.load_pipeline(name="en", tokenize_pretokenized=True)
    print("Models loaded, and they assume whitespace tokenized text")

    gold_sen, gold_ner = read_file(mypath)

    matches = 0
    mis_matches = 0

    spacy_netags = [] #will contain spacy preds
    stanza_netags = [] #will contain stanza preds

    for sen in gold_sen:
       actual_sen = " ".join(sen)
       doc_spacy = nlp(actual_sen)
       doc_stanza = snlp(actual_sen)
       temp_tags_spacy = []
       temp_tags_stanza = []

       for token in doc_spacy:
          if token.ent_iob_ and token.ent_type_:
            tag = token.ent_iob_+"-"+token.ent_type_
          else:
            tag = token.ent_iob_
          temp_tags_spacy.append(tag)

       for token in doc_stanza:
          if token.ent_iob_ and token.ent_type_:
            tag = token.ent_iob_+"-"+token.ent_type_
          else:
            tag = "O"

          temp_tags_stanza.append(tag)

       if temp_tags_spacy == temp_tags_stanza:
          matches = matches+1
       else:
          mis_matches = mis_matches+1

       spacy_netags.append(temp_tags_spacy)
       stanza_netags.append(temp_tags_stanza)

    print("***Basic stats: ****")
#    print("Num sentences: ", len(gold_sen), "in this genre: ", sys.argv[1].split("onto.")[1])
    print("Num. predictions where stanza and spacy match exactly: ", matches)
    print("Num. predictions where there is a difference between stanza and spacy: ", mis_matches)

    print("Classification report for Spacy NER: ")
    print(classification_report(gold_ner, spacy_netags, digits=4))

    print("Classification report for Stanza NER: ")
    print(classification_report(gold_ner, stanza_netags, digits=4))



def main():
    #Specify the path of the test file in bio format, and the function gives performance eval for spacy and stanza.
    mypath = "input_perturb/onto.all.test.perturb2.ner"
    spacy_and_stanza(mypath)

if __name__ == "__main__":
    main()
