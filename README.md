

1. Libraries:

 -- Spacy 3.0, with en_core_web_trf model
(instructions to install: https://spacy.io/usage)

 -- Stanza: check here - https://github.com/stanfordnlp/stanza-train. 

 -- Spark-NLP, version 3.1.2 is used. - https://nlp.johnsnowlabs.com/docs/en/install

2. spacy-stanza-scripts/ directory consists of the following information:
- scripts to convert bio files to spacy/stanza format (tool specific json)
- scripts to train and save ner models with spacy/stanza
- scripts to evaluate an existing ner model trained using train scripts above.
- config.cfg file for spacy.

3. Python files:
- conll-to-bio.py: converts ontonotes downloaded format from cemantrix and LDC to CONLL-BIO format, using a pre-existing script.
- evaluation_spacy_stanza.py: evaluates the NER models that come with spacy and stanza, on any test set in BIO format.
- evaluation_sparknlp.py: evaluates sparknlp's NER models
- generate_splits.py: generates 10 random train/dev/test splits for ontonotes dataset.
- input_perturbances_faker.py: generates new testsets for a given test set based on the chosen faker transformation. 
- train_sparknlp.py: script to train a NER model using SparkNLP's architecture. 
