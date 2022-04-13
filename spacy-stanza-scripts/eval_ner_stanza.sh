#run this from: stanza-train/stanza directory, after following the setup instructions on stanza-train github

for i in `seq 1 10`; 
do echo $i;
python3 -m stanza.models.ner_tagger --wordvec_pretrain_file ~/stanza_resources/en/pretrain/combined.pt --eval_file ~/stanza-ner-format-splits/fold$i"_test.ner" --lang en --shorthand en_fold$i --mode predict --save_name fold$i --cpu ;	
done	

