## Dataset
- Unzip data.zip to `./data`
    ```sh
    unzip data.zip -d ./data
    ```
- Folder structure
    ```
    .
    ├── data
    │   ├── test/
    │   └── train/
    ├── Fn.py
    ├── Predict.py
    ├── Readme.md
    ├── requirements.txt
    ├── Resize_video.py
    └── Train.py
    ```

## Environment
- torch 1.12.1+cu116 and torchaudio 0.12.1+cu116
    ```sh
    pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
    ```

- Python 3.6 or later version
    ```sh
    pip install -r requirements.txt
    ```




## Train
- To resize the videos train/test data into (256,256)
    ```sh
    python Resize_video.py
    ```
    The resized data will be saved in `./data_resized`

    - You can change your original data path, resized data path by: 
    ```sh
    python Train.py --ori_data_path= --path_to_save= 
    ```

- Train the model
    ```sh
    python Train.py
    ```
    The weights will be save in `./model/best_weights/bestweights.pth`

    - You can change your training data path, weights path(path to save trained path), cuda by:
    ```sh
    python Train.py --data_path= --weights_path= --CUDA=
    ```
    The training data need to be resized to (256,256) by `Resize_video.py` first!
    

## Make Prediction
```sh
python Predict.py
```

- You can change your testing data path, weights path(path to load trained weights), prediction path(path to dave the prediction), cuda by:
    ```sh
    python Train.py --data_path= --weights_path= --prediction_path = --CUDA=
    ```
    The training data need to be resized to (256,256) first!
The prediction file is `prediction.csv`.
