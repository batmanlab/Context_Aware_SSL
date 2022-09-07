# Perform registration to the atlas image. This will take a long time. 
# If you have many subjects, you can perform registration using the lung mask(https://github.com/JoHof/lungmask) rather than using the raw image.

import os
import SimpleITK as sitk
import argparse
import pandas as pd
import glob

def clamp_img(img_path, cif):
    # clip the intensity range for more robust registration
    img=sitk.ReadImage(img_path)
    img=cif.Execute(img)
    img=sitk.Cast(img, sitk.sitkInt16)
    sitk.WriteImage(img, "./INSP2Atlas/clamped/"+img_path.split('/')[-1])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--atlas_image', type=str)
    parser.add_argument('--input_csv', type=str)
    args = parser.parse_args()

    # Make output dirs
    os.makedirs("./INSP2Atlas/", exist_ok=True)
    os.makedirs("./INSP2Atlas/log", exist_ok=True)
    os.makedirs("./INSP2Atlas/clamped", exist_ok=True)
    os.makedirs("./INSP2Atlas/image", exist_ok=True)
    os.makedirs("./INSP2Atlas/transform", exist_ok=True)

    df = pd.read_csv(args.input_csv, sep=',')
    df_here = df[~df['image'].isnull()]
    moving_img_list = df_here['image'].tolist()

    # Ignore processed imgs
    img_list = glob.glob("./INSP2Atlas/transform/*.mat")
    finished_id = [os.path.basename(x).split("_")[0] for x in img_list]
    finished_id = set(finished_id)

    fixed_img = args.atlas_image
    LOG = open("./INSP2Atlas/log/ants_reg_idx"+str(args.batch_index)+".log", "w")

    # clamp image intensity
    cif=sitk.ClampImageFilter()
    cif.SetUpperBound(1024)
    cif.SetLowerBound(-1024)

    for idx, line in enumerate(moving_img_list):
        if line.split('/')[1] in finished_id:
            continue

        try:
            clamp_img(line, cif) # Clamp intensity
        except Exception as e: # Some images are corrupted
            print(e)
            print("Image loading error:", line)
            continue

        moving_img = "./INSP2Atlas/clamped/" + line.split('/')[-1]
        warped_img = os.path.basename(moving_img)[:-7] # Remove .nii.gz from file name
        
        # Main registration executive, you may change hyperparameter to adapt to your data
        run_result = os.system("antsRegistration -d 3 -o [./INSP2Atlas/transform/"+warped_img+"_Reg_Atlas_Affine_,./INSP2Atlas/image/"+warped_img+"_Reg_Atlas_Affine.nii.gz] -r ["+fixed_img+", "+moving_img+",1] -t Affine[0.01] -m MI["+fixed_img+", "+moving_img+",1,32,Regular,0.5] -c [500x250x100] -s 2x1x0 -f 4x2x1")

        # Only transform mat file is needed, transformed image is removed to save space
        if run_result == 0:
            try:
                os.unlink("./INSP2Atlas/image/"+warped_img+"_Reg_Atlas_Affine.nii.gz")
            except Exception as e:
                continue
        LOG.write(warped_img+"\t"+str(run_result)+"\n")

    LOG.close()
