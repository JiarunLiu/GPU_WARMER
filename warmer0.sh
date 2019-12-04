echo activate python envirouments
conda activate py37

echo setting cuda device cuda 0
export CUDA_VISIBLE_DEVICES=0

echo start progress
python WARMER.py
