#run this from the directory where all models should be saved 

for i in `seq 1 10`; 
do echo $i;
python3 -m spacy train config.cfg --paths.train ~/spacy-ner-format-splits/fold$i"_train.spacy" --paths.dev ~/spacy-ner-format-splits/fold$i"_dev.spacy" --output fold$i"_trfmodel" ;
done	

