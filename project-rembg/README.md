# Background Removal

Rembg is a tool to remove background from images. It uses deep learning models to detect the foreground and seperate it from the background. It can be used for various applications such as removing background from product images, portraits, and more. In my particular case, the reason for embarking on this project was to help out a friend who was working on a wadrobe management app that required the ability to remove the background from clothing images. The app would allow users to upload images of their clothes and then use the app to mix and match outfits. This is a python library that has cli and API support, making it easy to integrate into other applications.

## Instructions

HPCC Instructions:
```
module load Miniforge3
module load cuDNN
```

Create Conda environment:
```
conda create -n rembg_env python=3.11
```

1. Conda environment
```
conda activate rembg_env
```

2. Modify access rights to bash script:
```
chmod +x rembg.sh
```

3. Run the bash script on subset of images:
```
./rembg.sh
```
This will run the background removal on a subset of images from the original `images` folder and save the output in the `subset_output` folder. The code may take around `5 minutes to run because the subset of images consists over 50 images. 

If you want to test on an example image, from the root of the repository, you can run the following command:
```
rembg i output.png subset/18030.jpg
```

Timings:
- Time taken for 50 files with rembg: 267.58 seconds