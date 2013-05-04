DATA=/media/software/data/processed
#DATA=../data/raw
OUTFILE=../data/processed
SRC=../src

time python $SRC/subsample.py -n 70136 $DATA/onlyopen.csv > $DATA/equal_closed_open.csv
time sed 1d $DATA/onlyclosed.csv >> $DATA/equal_closed_open.csv 
time python $SRC/cut.py -l ReputationAtPostCreation,OwnerUndeletedAnswerCountAtPostTime,Title,BodyMarkdown,Tag1,Tag2,Tag3,Tag4,Tag5,OpenStatus $DATA/equal_closed_open.csv > $DATA/cutted_equal_closed_open.csv  
