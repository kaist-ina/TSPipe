# BYOL (Bootstrap Your Own Latent)
- Original code from https://github.com/sthalles/PyTorch-BYOL

## Configure
- Setup `config.yaml` : BYOL related configuration.
- Setup `tspipe.yaml` : TSPipe related configuration.

## How to run
```bash
python main.py --tspipe-config=tspipe.yaml --ip=localhost --rank=0 --num-nodes=1
```