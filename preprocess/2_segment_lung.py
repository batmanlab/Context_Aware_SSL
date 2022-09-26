import os
import argparse
import pandas as pd
import glob

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_csv', type=str)
    args = parser.parse_args()

    # Make output dirs
    os.makedirs("./tmp/", exist_ok=True)
    os.makedirs("./tmp/lung_segmentation", exist_ok=True)
    
    df = pd.read_csv(args.input_csv, sep=',')
    df_here = df[~df['image'].isnull()]
    moving_img_list = df_here['image'].tolist()

    for idx, img_path in enumerate(moving_img_list):

        # Main executive
        run_result = os.system("lungmask "+img_path+" ./tmp/lung_segmentation/"+img_path.split("/")[-1])