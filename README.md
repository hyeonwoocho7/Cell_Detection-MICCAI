# Cell_Detection-MICCAI
Official code of "Cell Detection in Domain Shift Problem Using Pseudo-Cell-Position Heatmap", Miccai 2021

## Directory Structure
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
