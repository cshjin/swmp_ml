#!/usr/bin/bash

# This script is used to generate the data for the paper
for network in uiuc150 epri21
do
  for efield_mag in 5 10 15 20
  do
    for efield_dir in 45 135
    do
      echo "gen $network $efield_mag/$efield_dir"
      python gen_gic_data.py --efield_mag $efield_mag --efield_dir $efield_dir --nums 50 --network $network  
    done
  done
done
