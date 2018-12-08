#!/bin/bash
#$ -M valenta4@uw.edu
#$ -l mem_requested=4G
#$ -l h_rt=08:00:00
#$ -o /net/gs/vol1/home/valenta4/silacDIA/eNo
#$ -e /net/gs/vol1/home/valenta4/silacDIA/eNo


# Set up environment for analysis
module load boost/1.49.0
module load glpk/4.47
module load qt/4.8.5
module load xerces-c/3.1.1
module load OpenMS/latest

module load python/3-anaconda
source activate msms

cd silacDIA

# Deploy analyses with a given penalty
echo "Starting analysis of ${exp} with a penalty of ${pen} and a secondary penalty of ${pen2}"
time python analyze_experiment.py ${exp} ${pen} ${pen2}