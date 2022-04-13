#run this from: stanza-train/stanza directory, after following the setup instructions on stanza-train github
#prepare ner data from bio to json, for all folds (10 folds, 3 files per fold -train, test, dev)
indir='/home/vajjalas/bio-splits/'
outdir='/home/vajjalas/stanza-ner-format-splits/'
mkdir $outdir
dir=`ls $indir`
for f in $dir; do
	echo $outdir$f
	python3 stanza/utils/datasets/prepare_ner_data.py $indir$f $outdir$f
done
