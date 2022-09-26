# Preprocess your data

**\*\*\* NEW \*\*\*** We release an all-in-one pipeline for feature extraction using our model pretrained on lung CT, available [here](https://drive.google.com/drive/folders/1ZdIuCo3uEZsxGj7drJ9SI7l48q0Ri3az?usp=sharing), sample data is provided.

The data preprocessing pipeline consists of five steps, listed as following:

### Step 1: Patchifying the atlas image & save the anatomical landmark locations
```bash
python 1_patchify_atlas.py --atlas_image path_to/atlas_image.nii.gz 
                           --atlas_roi_mask path_to/atlas_lung_mask.nii.gz
                           --output_dir ./patch_data_32_6_reg --patch_size 32 --step_size 26 
```
The `path_to/atlas_roi_mask.nii.gz` is the ROI mask for the atlas image, we use [lungmask](https://github.com/JoHof/lungmask) to segment lung region as ROI.
The script will print the number of patch for each subject, which will be used in step 4.

The atlas image we used for COPDGene (lung) dataset is available <a href="https://drive.google.com/file/d/1xNdrquyYRJthukQVZIWPKwMSbKPQccmp/view?usp=sharing">here</a>, and the output landmark location for lung CT is available <a href="https://drive.google.com/file/d/1SbuoCiN-_QZQTlQmWSzrs_jRCV1dtg76/view?usp=sharing">here</a>.

### Step 2: Lung segmentation
```bash
python 2_segment_lung.py --input_csv ./dataset.csv
```
The dataset.csv should at least contains two columns: sid and image, the sid column contains unique ID of subjects and the image column contains path to images of each subject.

### Step 3: Registration
```bash
python 3_registration.py --atlas_image ./misc/atlas_lung_mask.nii.gz \
                         --input_csv ./dataset.csv
```
We use registration on the lung mask for faster convergence and more robust performance.

### Step 4: Mapping landmarks and patchifying
```bash
python ./src/preprocess/4_patchify_images.py --atlas_image ./misc/atlas_lung_mask.nii.gz \
                            --atlas_patch_loc ./misc/atlas_patch_loc.npy \
                            --lowerThreshold -1024 --upperThreshold 240 \
                            --input_csv ./dataset.csv \
                            --output_dir ./results/processed_patch \
                            --num_processor 4 \
                            --patch_size 32 \
                            --step_size 26
```
The atlas\_patch\_loc.npy is the output patch location file from step 1.

### Step 5: Grouping patches (for pre-training only)
```
python 5_group_patch.py --num_patch 581
                        --batch_size 48
                        --num_jobs 28
                        --root_dir ./results/processed_patch/
```
The step is used to reduce IO demand and accelerate the training process.

After the five steps, the preprocessed dataset folder ./results/processed\_patch/ can be used for pre-training the model.
