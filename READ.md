## RSMPNet: Relationship Guided Semantic Map Prediction

This is a PyTorch implementation of the WACV24 paper, RSMPNet: Relationship Guided Semantic Map Prediction.

### Installing Dependencies

We install dependencies following L2M:

Create conda environment:

```
conda create -n habitat python=3.6 cmake=3.14.0
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
```

Installing habitat-sim and habitat-lab:

```` 
conda create -n habitat python=3.6 cmake=3.14.0
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch

git clone --branch v0.1.7 https://github.com/facebookresearch/habitat-sim.git
cd habitat-sim; pip install -r requirements.txt; python setup.py build_ext --parallel 2 install --headless --with-cuda

git clone --branch v0.1.7 https://github.com/facebookresearch/habitat-lab.git
cd habitat-lab; pip install --no-cache-dir h5py; pip install -r requirements.txt; python setup.py develop --all

pip install -r requirements.txt
````



### Setup

Clone the repository and install other requirements:



```
git clone 
```



Download semantic map prediction datasets [here](https://www.dropbox.com/scl/fi/4dpko4s8fhm1bj3lbx9ng/datasets.zip?rlkey=nuvpibd5cus5cioiqtk3v1fz2&dl=0).



### Usage

#### Training

```
python main.py --name smp_test --batch_size 1 --num_workers 1 --is_train --log_dir /disk2/Code/L2M/wacv_code/3070/ --stored_episodes_dir /home/jignwen/Code/MapPrediction/data/scene_datasets/mp3d/L2M/data_v5/mp3d_objnav_episodes_tmp/ --num_epochs 40
```



#### Downloading pre-trained models: 

You can download our pre-trained model [here](https://www.dropbox.com/scl/fo/annakmz80bh6kl2ztewqa/h?rlkey=vbgf71f6sfl2r516mso79bmic&dl=0).



#### Evaluation:

```
python main.py --name epoch48 --ensemble_dir ~/Code/L2M/semantic-map-prediction/trained_model/smp/ --log_dir /disk2/Code/L2M/wacv_code/3070/test/ --sem_map_test --stored_episodes_dir /home/jignwen/Code/MapPrediction/data/scene_datasets/mp3d/L2M/data_v5/mp3d_objnav_episodes_tmp/ 
```










