# Cell Detection for Domain Adaptation
This is Official code of "Cell Detection in Domain Shift Problem Using Pseudo-Cell-Position Heatmap", Miccai 2021

# Requirements
- Python3.7
- PyTorch
- OpenCV

# Train

## Example
```
$ python3 main.py --random_seed ${random_seed} --test_path ${test_path} --target_path ${target_path} --source_path ${source_path} --steps ${numbers of domain adaptation}
```


For implementation, your directory have to be structure like below.

<details>
<summary>Directory Structure</summary>


# Directory Structure
```
├── All_fscore.py
├── Data
│   ├── seq2
│   │   ├── gt
│   │   │   ├── seq2_00000_000_000.tif
│   │   │   ├── seq2_00000_000_001.tif
│   │   │   ├── seq2_00000_000_002.tif
│   │ 	│   │ 		.
│   │   │   │ 		.
│   │   │   │ 		.
│   │   │   │ 
│   │   │   ├── seq2_00099_007_006.tif
│   │   │   ├── seq2_00099_007_007.tif
│   │   │   ├── seq2_00099_007_008.tif
│   │   │   └── seq2_00099_007_009.tif
│   │   └── ori
│   │       ├── seq2_00000_000_000.tif
│   │       ├── seq2_00000_000_001.tif
│   │       ├── seq2_00000_000_002.tif
│   │       ├── seq2_00000_000_003.tif
			.
			.
			.

│   │       ├── seq2_00099_007_006.tif
│   │       ├── seq2_00099_007_007.tif
│   │       ├── seq2_00099_007_008.tif
│   │       └── seq2_00099_007_009.tif
│   └── test_seq6
│       ├── gt
│       │   ├── 00000.tif
│       │   ├── 00001.tif
│       │   ├── 00002.tif
		  .
		  .
		  .
│       
│       │   ├── 00098.tif
│       │   └── 00099.tif
│       └── ori
│           ├── 00000.tif
│           ├── 00001.tif
                  .
		  .
                  .
│           ├── 00098.tif
│           └── 00099.tif
├── Detection
│   ├── detection
│   │   ├── custom_loss.py
│   │   ├── detection_eval.py
│   │   ├── __init__.py
│   ├── fscore.py
│   ├── networks
│   │   ├── __init__.py
│   │   ├── network_model.py
│   │   ├── network_parts.py
│   ├── predict.py
│   ├── train.py
│   └── utils
│       ├── color.csv
│       ├── for_review.py
│       ├── __init__.py
│       ├── load.py
│       ├── matching.py
├── Discriminator
│   ├── Dataaugmentation.py
│   ├── distribution_sigmoid.py
│   ├── entropy_image_level.py
│   ├── eval.py
│   ├── __init__.py
│   ├── load.py
│   ├── predict.py
│   ├── resnet_dropout.py
│   ├── train.py
│   └── utils.py
├── main.py
├── Model
│   ├── Detection
│   │   └── step0
│   └── Discriminator
│       └── step0
```

</details>

## Citation
If you find the code useful for your research, please cite:
```
@inproceedings{cho2021cell,
  title={Cell detection in domain shift problem using pseudo-cell-position heatmap},
  author={Cho, Hyeonwoo and Nishimura, Kazuya and Watanabe, Kazuhide and Bise, Ryoma},
  booktitle={Medical Image Computing and Computer Assisted Intervention--MICCAI 2021: 24th International Conference, Strasbourg, France, September 27--October 1, 2021, Proceedings, Part VIII 24},
  pages={384--394},
  year={2021},
  organization={Springer}
}