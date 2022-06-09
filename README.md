
This repo contains the relevant code and result files for the paper ["What do we really know about State of the art NER?"](https://arxiv.org/abs/2205.00034), to appear in LREC 2022, co-authored by Sowmya Vajjala (National Research Council, Canada) and Ramya Balasubramaniam (Novisto, Canada). 

**Here is some information on the core NLP libraries and their versions:**  
 -- Spacy 3.0, with en_core_web_trf model
(instructions to install: https://spacy.io/usage)

 -- Stanza: check here - https://github.com/stanfordnlp/stanza-train. 

 -- Spark-NLP, version 3.1.2 is used. - https://nlp.johnsnowlabs.com/docs/en/install

**Here is some description about the rest of the contents in this repo:**  
spacy-stanza-scripts/ directory consists of the following information:
- scripts to convert bio files to spacy/stanza format (tool specific json)
- scripts to train and save ner models with spacy/stanza
- scripts to evaluate an existing ner model trained using train scripts above.
- config.cfg file for spacy.

Python files:
- conll-to-bio.py: converts ontonotes downloaded format from cemantrix and LDC to CONLL-BIO format, using a pre-existing script.
- evaluation_spacy_stanza.py: evaluates the NER models that come with spacy and stanza, on any test set in BIO format.
- evaluation_sparknlp.py: evaluates sparknlp's NER models
- generate_splits.py: generates 10 random train/dev/test splits for ontonotes dataset.
- input_perturbances_faker.py: generates new testsets for a given test set based on the chosen faker transformation. 
- train_sparknlp.py: script to train a NER model using SparkNLP's architecture. 

Results-Details.xlsx contains all the detailed result tables (and the figures generated out of those). 
