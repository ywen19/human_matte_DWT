# Human Matting via Discrete Wavelet Transform

We build a frequency-centric matting framework that places frequency-derived features at the center of the modeling 
pipeline for alpha matting task. 

The dataset we used is derived from 
[Background Matting 240K](https://grail.cs.washington.edu/projects/background-matting-v2/#/datasets).


## 1. Environment Configuration
To configure the environment:
```
chmod +x env_setup.sh
./env_setup.sh
conda activate cuda118
```

## 2. Data Preprocess
The VideoMatte240K dataset provides 484 pairs of high-resolution alpha matte and foreground video clips, constituting 
240,709 unique frames. The alpha matte and foreground data were extracted from green screen stock footage. 
The dataset only provides foreground layers(human foreground) and associated alpha masks. 

To follow nowadays paradigm, we apply Reihard color transfer when composite FG with random BG.

To compose human rgb foregrounds with random background images, 
```
cd preprocess
python compose_video_bg.py
```

## 3. Framewise Data Arrangement and Coarse Guide Mask Generation
Although our model is guide(auxiliary-free), we use some works that utilize guide masks as semantic hints for comparison. 
To extract the human foreground coarse mask as a semantic guide for our framework, we use YOLOV8+SAM2. 
Given 'person' token to 'YOLOV8', it will return the bounding boxes for all human subjects in a given frame.
We then use these bounding boxes as input to SAM2. 

First, we need to arrange our dataset from original video format to per-frame .png.
The dataset is arranged as:
```
<data_folder>
     | -- test
            | -- alpha
                   | -- <video_name>
                            | -- frames
                                   | -- 0000.png
                                   | -- ...
                            | -- video
                                   | -- <video_name>.mp4
            | -- fgr
                   | -- <video_name>
                            | -- frames
                                   | -- 0000.png
                                   | -- ...
                            | -- video
                                   | -- <video_name>.mp4
     | -- train
            | -- alpha
                   | -- <video_name>
                            | -- frames
                                   | -- 0000.png
                                   | -- ...
                            | -- video
                                   | -- <video_name>.mp4
            | -- fgr
                   | -- <video_name>
                            | -- frames
                                   | -- 0000.png
                                   | -- ...
                            | -- video
                                   | -- <video_name>.mp4

```
To convert such data arrangement,
```
chmod +x run_frame_extraction.sh
RATIO=0.5 ./run_frame_extraction.sh  
```
where 0.5 (means 50%) is the sample ratio of frames/total_frames for each video. 
T his is to help us randomly sample data subsets.

Then we could apply coarse mask extraction:
```
chmod +x run_mask_extraction.sh
./run_mask_extraction.sh first_frame or ./run_mask_extraction.sh all_frames
```
The argument basically allows us to extract the guide mask on all the video frames or just the first frame for 
each video.

## 3. Training/Evaluation
Under the `pipeline` directory, `solver_plateau.py`, `inference.py` are the ones we used to run 
training/validation/inference.

To run training and validation,
```
cd pipeline
python solver_plateau.py
```

To run inference,
```
cd pipeline
python inference.py --image <image_path or folder path for multiple images> 
                    --checkpoint <checkpoint_path> 
                    --outdir <output_path> 
                    --save-overlay(if want to save predicted FG on a green background)
```
The inference code also supports other arguments. Please refer to the code for more options.
