#run this from: stanza-train/stanza directory, after following the setup instructions on stanza-train github

for i in `seq 1 10`; 
do echo $i;
python3 -m stanza.models.ner_tagger --wordvec_pretrain_file ~/stanza_resources/en/pretrain/combined.pt --train_file ~/stanza-ner-format-splits/fold$i"_train.ner" --eval_file ~/stanza-ner-format-splits/fold$i"_dev.ner" --lang en --shorthand en_fold$i --mode train --save_name fold$i --cpu --max_steps 5000 ;	
done	

