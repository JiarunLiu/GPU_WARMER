

for gpu_id in "$@"; do
  python WARMER.py -b 24 --mode maximum_single -gid ${gpu_id} &
  python WARMER.py -b 24 --mode maximum_single -gid ${gpu_id} &
done