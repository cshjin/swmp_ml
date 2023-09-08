#!/usr/bin/bash

# This script is used to generate the data for the paper
python gen_gic_data.py --efield_mag 5 --efield_dir 45 --nums 50 --network epri21
echo "gen 10/45"
python gen_gic_data.py --efield_mag 10 --efield_dir 45 --nums 50 --network epri21
echo "gen 15/45"
python gen_gic_data.py --efield_mag 15 --efield_dir 45 --nums 50 --network epri21
echo "gen 20/45"
python gen_gic_data.py --efield_mag 20 --efield_dir 45 --nums 50 --network epri21
echo "gen 5/135"
python gen_gic_data.py --efield_mag 5 --efield_dir 135 --nums 50 --network epri21
echo "gen 10/135"
python gen_gic_data.py --efield_mag 10 --efield_dir 135 --nums 50 --network epri21
echo "gen 15/135"
python gen_gic_data.py --efield_mag 15 --efield_dir 135 --nums 50 --network epri21
echo "gen 20/135"
python gen_gic_data.py --efield_mag 20 --efield_dir 135 --nums 50 --network epri21