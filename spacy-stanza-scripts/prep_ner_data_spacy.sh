# Apr 2021: Script to convert BIO format splits to Spacy's .spacy.
# Note: This is different from previous spacy version, which used .json format. 
home="bio-splits/"
output="spacy-ner-format-splits/"
temp=`ls $home$i`
for j in $temp; do
   python3 -m spacy convert $home$j $output -c ner
done
