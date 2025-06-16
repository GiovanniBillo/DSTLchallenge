import os
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

N_CLS=10 # different types of object classes to distinshish in image segmentation 

RAW_DATA_DIR = os.path.join(DATA_DIR, 'dstl-satellite-imagery-feature-detection')
DF = pd.read_csv(RAW_DATA_DIR+ '/train_wkt_v4.csv')
GS = pd.read_csv(RAW_DATA_DIR+ '/grid_sizes.csv', names=['ImageId', 'Xmax', 'Ymin'], skiprows=1)
SB = pd.read_csv(os.path.join(RAW_DATA_DIR, 'sample_submission.csv'))
ISZ = 160
smooth = 1e-12

IN_CHANS=8
PRETRAINED_VIT_PATH='google/vit-base-patch16-224-in21k'
