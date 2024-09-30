## Dataset
- Unzip data.zip to `./data`
    ```sh
    unzip data.zip -d ./data
    ```
- Folder structure
    ```
    ./
    ├── data
    │   ├── unlabeled/
    │   └── test/
    ├── Readme.md
    ├── requirements.txt
    ├── dataset.py
    ├── embedding.py
    ├── lossFn.py
    ├── metric.py
    ├── model.py
    ├── test.py
    ├── train.py
    ├── transforms.py
    ```

## Environment
- Python 3.6 or later version
    ```sh
    pip install -r requirements.txt
    ```

## Train
```sh
python train.py --logdir path/to/logs
```
The best weights will be save in `./model/best_weights/bestweights.pth`
- You can change your training weights path(path to save trained path), cuda by:
```sh
python train.py  --weights_path= --device=
```

## Test
```sh
python test.py 
```
- You can change your trained weights path(path of trained path), cuda by:
```sh
python test.py  --weights_path= --device=
```

## Get the embedding of unlabeled data
```sh
python embedding.py 
```
The embedding .npy file will be saved in `./embedding.npy`
- You can change your trained weights path(path of trained path), cuda, embedding path(path to save the embedding .npy file) by:
```sh
python test.py  --weights_path= --device= --embedding_path=
```