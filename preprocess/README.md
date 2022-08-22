# Preprocess your data

The data preprocessing pipeline consists of four steps, including patchifying the atlas image, registration, mapping landmarks and patchifying, grouping patches.

### Step 1: Patchifying the atlas image
```bash
python 1_patchify_atlas.py --atlas_image path_to/atlas_image.nii.gz 
                           --atlas_roi_mask path_to/atlas_roi_mask.nii.gz
                           --output_dir ./patch_data_32_6_reg --patch_size 32 --step_size 26 
```
The `path_to/atlas_roi_mask.nii.gz` is the ROI mask for the atlas image, we use [lungmask](https://github.com/JoHof/lungmask) to segment lung region as ROI.
The script will print the number of patch for each subject, which will be used in step 4.

The atlas image we used for COPDGene (lung) dataset is available <a href="https://drive.google.com/file/d/1xNdrquyYRJthukQVZIWPKwMSbKPQccmp/view?usp=sharing">here</a>.

### Step 2: Registration
```bash
python 2_registration.py --atlas_image path_to/atlas_image.nii.gz
                         --input_csv path_to/dataset.csv
```
The dataset.csv should at least contains two columns: sid and image, the sid column contains unique ID of subjects and the image column contains path to images of each subject.

### Step 3: Mapping landmarks and patchifying
```bash
python 3_patchify_images.py --atlas_image path_to/atlas_image.nii.gz
                            --atlas_patch_loc ./patch_data_32_6_reg/atlas_patch_loc.npy
                            --lowerThreshold -1024 --upperThreshold 240
                            --input_csv path_to/dataset.csv
                            --output_dir ./patch_data_32_6_reg
                            --num_processor 4
                            --patch_size 32
                            --step_size 26
```
The atlas\_patch\_loc is the output patch location file from step 1.

### Step 4: Grouping patches
```
python 4_group_patch.py --num_patch num_patch
                        --batch_size 48
                        --num_jobs 28
                        --root_dir ./patch_data_32_6_reg/
```
The step is used to reduce IO demand and accelerate the training process.

After the four steps, the preprocessed dataset folder ./patch\_data\_32\_6\_reg/ can be used for training the model.
