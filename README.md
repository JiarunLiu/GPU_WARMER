# GPU_WARMER

Using GPU for heating. 

## Requirements:

- Python>=3.6
- pytorch>=0.4
- torchvision
- Bumpy

## Usage

warmer0.sh/warmer1.sh is using for GPU0/GPU1.

```
sh warmer0.sh
```

or using python script directly

```
export CUDA_VISIBLE_DEVICE=0
python WARMER.py
```

