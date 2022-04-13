#Use FAKER python library to generate new test sets with perturbed entities
import csv
import pandas as pd
from faker import Faker
import random

mypath = "onto.test.ner"

#anything to edit
mycat = 'PERSON' #PERSON, ORG, GPE
#GPE is countries, cities, states; ORG: companies, agencies, institutions;
fake = Faker('en_IN') #This can be changed to many other options. 
myoutput = "perturb_en-in_f.ner" #CHANGE THIS

fh = open(mypath)
fw = open(myoutput, "w", encoding="utf-8")

numcols = 4
for line in fh:
    splits = line.strip().split("\t")
    if len(splits) == 4:
        if splits[3] == 'B-'+mycat:
            #Lines commented below are for GPE. 
            splits[0] = fake.name_female().split()[0]
            #splits[0] = random.choice([fake.county(), fake.country(), fake.city()]).split()[0]
        elif splits[3] == 'I-'+mycat:
            splits[0] = fake.last_name_female().split()[0]
            #splits[0] = fake.city_suffix().split()[0]
        print("\t".join(splits))
        fw.write("\t".join(splits))
        fw.write("\n")
    else:
        fw.write("\n")

fh.close()
fw.close()
print("DONE")

