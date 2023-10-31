## RSMPNet: Relationship Guided Semantic Map Prediction

Jingwen Sun, Jing Wu, Ze Ji, Yu-Kun Lai

Winter Conference on Applications of Computer Vision (WACV) 2024

### Installing Dependencies

Create conda environment:

```
conda create -n habitat python=3.6 cmake=3.14.0
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
```

Installing habitat-sim and habitat-lab:

```` 
git clone --branch v0.1.7 https://github.com/facebookresearch/habitat-sim.git
cd habitat-sim; pip install -r requirements.txt
python setup.py build_ext --parallel 2 install --headless --with-cuda

git clone --branch v0.1.7 https://github.com/facebookresearch/habitat-lab.git
cd habitat-lab; pip install -r requirements.txt 
python setup.py develop --all
````



### Setup

Clone the repository and install other requirements:

```
git clone https://github.com/jws39/semantic-map-prediction.git
cd semantic-map-prediction; pip install -r requirements.txt
```

Download semantic map prediction datasets [here](https://www.dropbox.com/scl/fi/4dpko4s8fhm1bj3lbx9ng/datasets.zip?rlkey=nuvpibd5cus5cioiqtk3v1fz2&dl=0), and put the folder `smp` in in the following format:

```
your-datasets-path/
  mp3d_objnav_episodes_tmp/
    train/
      1LXtFkjw3qL/
        ep_1_40970_1LXtFkjw3qL.npz
        ...
    val/
      VVfe2KiqLaN/
        ep_1_16987_VVfe2KiqLaN.npz
        ...
    test/
      2azQ1b91cZZ/
        ep_1_1_2azQ1b91cZZ.npz
        ...
```



### Usage

#### Training

```
python main.py --name smp_test --batch_size 1 --num_workers 1 --is_train --log_dir you-log-path --stored_episodes_dir you-datasets-path/mp3d_objnav_episodes_tmp/ --num_epochs 50
```



#### Downloading pre-trained models: 

You can download our pre-trained model [here](https://www.dropbox.com/scl/fo/z2kj03w1eq86sx33n91h2/h?rlkey=6wlpclxzblollb4wyhkf8i6a2&dl=0), and put the folder `smp` in in the following format:

```
your-model-path/
  smp/
    model1/
      smp.pt
```

#### Evaluation:

```
python main.py --name exp_name --ensemble_dir your-model-path/smp/ --log_dir your-log-dir --sem_map_test --stored_episodes_dir you-datasets-path/mp3d_objnav_episodes_tmp/ 
```

## Related Projects

- This project builds on the [Learning to Map for Active Semantic Goal Navigation](https://github.com/ggeorgak11/L2M#learning-to-map-for-active-semantic-goal-navigation) paper and [Learning-Semantic-Associations-for-Mirror-Detection](https://github.com/guanhuankang/Learning-Semantic-Associations-for-Mirror-Detection) paper.



