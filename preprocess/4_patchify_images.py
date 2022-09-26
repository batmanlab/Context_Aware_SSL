'''
Pre-process the data to extract patches
Input: A csv file containing path to input files

'''
import argparse
import os
import sys
import math
import numpy as np
import SimpleITK as sitk
import pandas as pd
import multiprocessing as mp
import glob

def convert_to_isotropic(inputVolume, isoSpacing=1.0):

    inputSpacing = inputVolume.GetSpacing()
    inputSize = inputVolume.GetSize()
    #Resample the images to make them iso-tropic
    resampleFilter = sitk.ResampleImageFilter()
    T = sitk.Transform()
    T.SetIdentity()
    resampleFilter.SetTransform(T)
    resampleFilter.SetInterpolator(sitk.sitkBSpline)
    resampleFilter.SetDefaultPixelValue(float(-1024))
    # isoSpacing = 1 #math.sqrt(inputSpacing[2] * inputSpacing[0])
    resampleFilter.SetOutputSpacing((isoSpacing,isoSpacing,isoSpacing))
    resampleFilter.SetOutputOrigin(inputVolume.GetOrigin())
    resampleFilter.SetOutputDirection(inputVolume.GetDirection())
    dx = int(inputSize[0] * inputSpacing[0] / isoSpacing)
    dy = int(inputSize[1] * inputSpacing[1] / isoSpacing)
    dz = int((inputSize[2] - 1 ) * inputSpacing[2] / isoSpacing)
    resampleFilter.SetSize((dx,dy,dz))
    try:
        resampleVolume = resampleFilter.Execute(inputVolume)
    except Exception as err:
        print("Resample failed: " + str(imageFilePath) )
        print(err.decode(encoding='UTF-8'))
        return None

    return resampleVolume

def pad_img(input_img, image_lowest_intensity=-1024):
    lower_bound = [60] * 3
    upper_bound = [60] * 3
    cpf = sitk.ConstantPadImageFilter()
    cpf.SetConstant(image_lowest_intensity)
    cpf.SetPadLowerBound(lower_bound)
    cpf.SetPadUpperBound(upper_bound)
    input_img = cpf.Execute(input_img)
    return input_img

def Image2Patch(inputImg, step_size, patch_size, registered_patch_loc):
    """ This function converts image to patches. 
        Here is the input of the function:
          inputImg : input image. This should be simpleITK object
          patchSize : size of the patch. It should be array of three scalar
        Here is the output of the function:
          patchImgData : It is a list containing the patches of the image
          patchLblData : Is is a list containing the patches of the label image
          
    """
    patch_vol = patch_size[0]*patch_size[1]*patch_size[2]
    patch_img_data = []
    patch_loc = []

    for i in range(registered_patch_loc.shape[0]):
        x, y, z = registered_patch_loc[i].tolist()
        #print(x,y,z)
        patchImg = sitk.RegionOfInterest(inputImg, size=patch_size, index=[x,y,z])
        npLargePatchImg = sitk.GetArrayFromImage(patchImg)
        patch_img_data.append(npLargePatchImg.copy())
        patch_loc.append([x, y, z])
    
    patch_img_data = np.asarray(patch_img_data)
    patch_loc = np.asarray(patch_loc)
    return patch_img_data, patch_loc

def extract_patch(isoRawImage_file, altas_patch_loc):
    #Read the input isotropic image volume
    isoRawImage = sitk.ReadImage(isoRawImage_file)
    isoRawImage = convert_to_isotropic(isoRawImage)
    isoRawImage = pad_img(isoRawImage)
    npIsoRawImage = sitk.GetArrayFromImage(isoRawImage)
    #print(npIsoRawImage.shape)

    # Thresholding the isotropic raw image
    npIsoRawImage[npIsoRawImage > upperThreshold] = upperThreshold
    npIsoRawImage[npIsoRawImage < lowerThreshold] = lowerThreshold

    thresholdIsoRawImage = sitk.GetImageFromArray(npIsoRawImage)
    thresholdIsoRawImage.SetOrigin(isoRawImage.GetOrigin())
    thresholdIsoRawImage.SetSpacing(isoRawImage.GetSpacing())
    thresholdIsoRawImage.SetDirection(isoRawImage.GetDirection())
    
    # Prepare registered patch location
    registered_patch_loc = []
    affine_trans=sitk.ReadTransform("./results/transform/"\
        + isoRawImage_file.split('/')[-1][:-7]+"_Reg_Atlas_Affine_0GenericAffine.mat")
    for i in range(altas_patch_loc.shape[0]):
        physical_cor_on_fixed = tuple(altas_patch_loc[i])
        physical_cor_on_moving = affine_trans.TransformPoint(physical_cor_on_fixed)
        index_on_moving = isoRawImage.TransformPhysicalPointToIndex(physical_cor_on_moving)
        registered_patch_loc.append(list(index_on_moving))
    registered_patch_loc = np.array(registered_patch_loc)

    #Extract Patches
    # Generate Patches of the masked Image
    patchImgData, patch_loc = Image2Patch(thresholdIsoRawImage, \
            [step_size]*3, [patch_size]*3, registered_patch_loc)
    return patchImgData, patch_loc

def prep_adjacency_matrix(patch_loc):
    adj = []
    for i in range(patch_loc.shape[0]):
        adj_row = np.zeros((patch_loc.shape[0],))
        dist = np.abs(patch_loc - patch_loc[i])
        max_side_dist = dist.max(1)
        dist = dist[max_side_dist<patch_size,:]
        volume = np.abs(dist-patch_size)
        volume = volume[:,0] * volume[:,1] * volume[:,2]
        #print(volume.shape)
        #print(adj_row[max_side_dist<patch_size].shape)
        adj_row[max_side_dist<patch_size] = volume / (patch_size**3)
        adj.append(adj_row.transpose())
    adj = np.asarray(adj)
    #adj = (adj / np.sum(adj, 0)).transpose()
    return adj

def run(start, end): 
    df = pd.read_csv(input_csv)
    df = df[~df['image'].isnull()]

    # Prepare physical coord of patch location
    altas_patch_loc = np.load(atlas_patch_loc_path)
    fixed_img = sitk.ReadImage(atlas_image_path)
    altas_patch_loc_temp=[]
    for i in range(altas_patch_loc.shape[0]):
        altas_patch_loc_temp.append(list(fixed_img.TransformIndexToPhysicalPoint(tuple(altas_patch_loc[i,:].tolist()))))
    altas_patch_loc=np.array(altas_patch_loc_temp)
    del altas_patch_loc_temp, fixed_img

    for i in range(start,end):
        row = df.iloc[i]   
        subject_id = row['sid']
        #if subject_id != '19676E':
        #    continue
        print("Processing", row['image'])
        output_basename = subject_id
        isotropicFileName = row['image']
        patchFileName = os.path.join(output_dir, 'patch', output_basename+'_patch.npy')
        if os.path.exists(patchFileName):
            continue
        if not os.path.exists(isotropicFileName):
            print(output_basename, "image not found")
            continue
        if not os.path.exists("./results/transform/"\
            + output_basename +"_Reg_Atlas_Affine_0GenericAffine.mat"):
            print(output_basename, "mat not found")
            continue

        if not os.path.exists(patchFileName):
            try:
                patchImgData, patch_loc = extract_patch(isotropicFileName, altas_patch_loc)

                adj = prep_adjacency_matrix(patch_loc)
                np.save(patchFileName, patchImgData)
                np.save(os.path.join(output_dir, 'patch_loc', output_basename+'_patch_loc.npy'), patch_loc)
                np.save(os.path.join(output_dir, 'adj', output_basename+'_adj.npy'), adj)
            except Exception as e:
                print(e)
                print("Failed in extract patch: " + str(output_basename) )
                continue
    return
                
def main(argv): 
    global data_dir
    global output_dir
    global input_csv
    global patch_size 
    global step_size
    global atlas_patch_loc_path
    global atlas_image_path
    global lowerThreshold, upperThreshold

    #Parse the arguments
    parser = argparse.ArgumentParser(description='Subject2Vector Data Preprocessing')
    parser.add_argument('-i', '--input_csv', type=str,\
                        default='/pghbio/dbmi/batmanlab/Data/COPDGene/Database/Final_Status_Phase-1_18_11_2019_13_12_04.csv',\
                        help = 'Input csv with patient_id, isotropic volume path and lung segmentation path.')
    parser.add_argument('--atlas_image', type=str, default="19676E_INSP_STD_JHU_COPD_BSpline_Iso1.0mm.nii.gz")
    parser.add_argument('--atlas_patch_loc', type=str, \
                        default="./patch_data_32_6_reg/19676E_INSP_STD_JHU_COPD_BSpline_Iso1_patch_loc.npy")
    parser.add_argument('-d', '--data_dir', type=str, default='/pghbio/dbmi/batmanlab/Data/COPDGene/',\
                        help='Directory where input data is stored. There should be one folder for each patient id in input csv')
    parser.add_argument('-o', '--output_dir', type=str, default='./patch_data_32_6_reg_mask/',\
                        help='Directory where intermediate and final files are saved.')
    parser.add_argument('-p', '--num_processor', type=int, default=16,\
                        help='Preprocess files in parallel. 1: no-parallel, n: number of nodes for parallel use')
    parser.add_argument('-s', '--patch_size', type=int, default=32, help='The size of the 3D patch.')
    parser.add_argument('-l', '--step_size', type=int, default=26, help='The overlap between consecutive patches.')
    parser.add_argument('--lowerThreshold', type=int, default=-1024)
    parser.add_argument('--upperThreshold', type=int, default=240)
    
    
    args = parser.parse_args()
    input_csv = args.input_csv
    data_dir = args.data_dir
    output_dir = args.output_dir
    num_processor = args.num_processor
    patch_size = args.patch_size
    step_size = args.step_size
    atlas_patch_loc_path = args.atlas_patch_loc
    atlas_image_path = args.atlas_image
    lowerThreshold = args.lowerThreshold
    upperThreshold = args.upperThreshold
    
    # make output dir
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'patch'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'patch_loc'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'adj'), exist_ok=True)

    df = pd.read_csv(input_csv)
    df = df[~df['image'].isnull()]

    start = 0
    end = df.shape[0]

    if num_processor > 1:
        processes = []
        count = math.ceil((end - start)/num_processor)
        for i in range(num_processor):
            if start + count > end:
                count = end - start
            processes.append(mp.Process(target=run, args=(start, start + count)))
            start += count
        for p in processes:
            p.start()
    else:
        run(start, end)

if __name__ == '__main__':
    main(sys.argv[1:])