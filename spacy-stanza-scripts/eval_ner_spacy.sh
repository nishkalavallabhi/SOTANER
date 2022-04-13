#run this from the directory where all models are saved

for i in `seq 1 10`; 
do echo $i;
python3 -m spacy evaluate fold$i"_model/model-best/" ~/spacy-ner-format-splits/fold$i"_test.spacy" --output fold$i"_eval" -G ;
done	

