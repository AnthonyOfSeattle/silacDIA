#!/bin/bash


experiments=(UWPROFL0360 UWPROFL0361 UWPROFL0362 UWPROFL0363 UWPROFL0364)
penalties=(1e5 3e5 1e6 3e6 1e7 3e7 1e8 3e8 1e9 3e9 1e10 3e10 1e11 3e11 1e12 3e12 1e13 3e13)
secondary_penalties=(1e7 1e8 1e9 1e10 1e11)
for exp in ${experiments[@]}
do
    for pen in ${penalties[@]}
    do
        for pen2 in ${secondary_penalties[@]}
        do
            qsub -N "silac_${exp}_${pen}_${pen2}" -v pen=${pen} -v pen2=${pen2} -v exp=${exp} ./handler.sh
        done
    done
done