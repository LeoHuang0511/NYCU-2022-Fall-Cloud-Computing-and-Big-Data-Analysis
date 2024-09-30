## Dataset
- Unzip mnist.zip to `./`
    ```sh
    unzip mnist.zip -d ./
    ```
- Save mnist.npz to `./`

- Folder structure
    ```
    ./
    ├── mnist
    ├── mnist.npz
    ├── Readme.md
    ├── requirements.txt
    ├── functions.py
    ├── generate_image.py
    ├── metrics_core.py
    ├── metric.py
    ├── models.py
    ├── train.py
    ```

## Environment
- Python 3.6 or later version
    ```sh
    pip install -r requirements.txt
    ```

## Train
```sh
python train.py 
```
The best weights will be save in `./weights/_batch64_lr0.0005_T500/best.pth`


## Generate Images
```sh
python generate_image.py 
```
The generative images will be save in `./final_generated_images/_batch64_lr0.0005_T500/images`
The diffusion process will be save in `./final_generated_images/_batch64_lr0.0005_T500/diffusion_process/diffusion_process.png`



