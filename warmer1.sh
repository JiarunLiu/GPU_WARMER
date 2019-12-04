echo activate python envirouments
conda activate py37

echo setting cuda device cuda 1
export CUDA_VISIBLE_DEVICES=1

echo start progress
python WARMER.py
