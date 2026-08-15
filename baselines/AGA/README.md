# Action-Guided Attention for Video Action Anticipation (under construction)

This repository contains the code for the paper **Action-Guided Attention for Video Action Anticipation**. 

## Prepare the Dataset
### Download the dataset

To download the EPIC-Kitchens-100 dataset, please refer to the [EPIC-KITCHENS download scripts](https://github.com/epic-kitchens/epic-kitchens-download-scripts), executing with the following commands:

```bash
cd preprocessing
# Pass the path to the epic-kitchens-download-scripts repository as the first argument
sh download.sh /path/to/epic-kitchens-download-scripts
```

### Untar the Dataset and build LMDB format

Extract the downloaded `.tar` files using the `untar.sh` script:

```bash
# Pass the target directory where the .tar files are located
sh untar.sh ./ek100
```

Once the frames are extracted, construct the LMDB dataset using:

```bash
sh build_lmdb.sh
```

## Model Inference

Use `inference.py` to load model weights and perform inference:

```bash
python inference.py --checkpoint <path_to_checkpoint> --in_dim 1024 --hidden_dim 2048 --out_dim 3806 --order 30
```

You can view the full list of arguments by running `python inference.py --help`.

## HD-EPIC Streaming baseline (this repo)

The original scripts target EPIC-Kitchens-100 LMDB features. For **HD-EPIC P01
Streaming Video** action anticipation at **+2 / +4 / +6 s**, use:

```bash
sbatch /home/ll5914/Jepa_baseline/aga_hdepic/submit_aga_p01_stream.slurm
```

See [`../aga_hdepic/README.md`](../aga_hdepic/README.md).

## Model Training

Use `train.py` to train the AGA model. It expects an LMDB dataset by default (handled internally through the `EK100Dataset` class).

```bash
python train.py --batch_size 128 --epochs 100 --lr 0.001 --weight_decay 0.01 --data_dir /path/to/lmdb
```

*(Note: Adjust the parameters inside or via command line based on your training configuration.)*

## Cite this Paper

If you find this code useful, please consider citing our paper:

```bibtex
@inproceedings{tai2026action,
  title={Action-Guided Attention for Video Action Anticipation},
  author={Tai, Tsung-Ming and Casarin, Sofia and Pilzer, Andrea and Nutt, Werner and Lanz, Oswald},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2026},
  url={https://openreview.net/forum?id=uKFVZMPppq}
}
```