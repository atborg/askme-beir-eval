
# BEIR evaluation for AskMe output

## requirements

1. use Python 3.7
2. score output file from AskMe in TSV format with the following columns - query id, document id, score  
(see example.tsv for formatting)

## steps to run evaluation

1. open eval.py to edit the following variables at the top of the script
2. change "dataset" variable to your desired BEIR [dataset](https://github.com/UKPLab/beir#beers-available-datasets)
3. change "filePath" variable to your AskMe output file
4. run the script!
