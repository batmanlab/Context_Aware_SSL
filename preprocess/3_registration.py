# Perform registration to the atlas image. This will take a long time. 

import os
import SimpleITK as sitk
import argparse
import pandas as pd
import glob

def clamp_img(img_path): # preprocessing for more robust registration
    img=sitk.ReadImage(img_path)
    img_arr=sitk.GetArrayFromImage(img)
    img_arr[img_arr==1]=50
    img_arr[img_arr==2]=100
    new_img=sitk.GetImageFromArray(img_arr)
    new_img.CopyInformation(img)
    new_img=sitk.Cast(new_img, sitk.sitkInt16)
    sitk.WriteImage(new_img, "./tmp/clamped/"+img_path.split('/')[-1])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--atlas_image', type=str)
    parser.add_argument('--input_csv', type=str)
    args = parser.parse_args()

    # Make output dirs
    os.makedirs("./results/", exist_ok=True)
    os.makedirs("./results/transform", exist_ok=True)
    os.makedirs("./tmp/clamped", exist_ok=True)
    os.makedirs("./tmp/image", exist_ok=True)
    
    df = pd.read_csv(args.input_csv, sep=',')
    df_here = df[~df['image'].isnull()]
    moving_img_list = df_here['image'].tolist()

    fixed_img = args.atlas_image

    for idx, line in enumerate(moving_img_list):
        moving_img = "./tmp/clamped/" + line.split('/')[-1]
        warped_img = os.path.basename(moving_img)[:-7] # remove .nii.gz

        try: # Use try because some images are corrupted
            clamp_img("./tmp/lung_segmentation/" + line.split('/')[-1]) # Clamp intensity
        except Exception as e:
            print(e)
            print("Image loading error:", line)
            continue

        # Main executive
        print("Processing", line)
        run_result = os.system("antsRegistration -d 3 -o [./results/transform/"+warped_img+"_Reg_Atlas_Affine_,./tmp/image/"+warped_img+"_Reg_Atlas_Affine.nii.gz] -r ["+fixed_img+", "+moving_img+",1] -t Affine[0.01] -m MI["+fixed_img+", "+moving_img+",1,32,Regular,0.5] -c [500x250x100] -s 2x1x0 -f 4x2x1")

        if run_result == 0: # only need transform matrix, delete tranformed image to save space
            try:
                os.unlink("./tmp/image/"+warped_img+"_Reg_Atlas_Affine.nii.gz")
            except Exception as e:
                continue