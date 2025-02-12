# HSTGNN
This is the pytorch implementation of HSTGNN. 

This paper has been accepted by [Information fusion 2025.](https://doi.org/10.1016/j.inffus.2025.102978)


## Dependencies
The code is built based on Python 3.8. Run the following code to install packages.

```
pip install tables pandas scipy Pywavelets
```

## Datasets
The datasets can be download from [https://github.com/Davidham3/STSGCN], [https://github.com/liyaguang/DCRNN] and [https://github.com/yuyolshin/SeoulSpeedData]. We provide splited dataset and you can also download processed data from [here](https://pan.quark.cn/s/5ed91f4d60b5).

## Train Commands
Run the following code to train the model.

```
python train.py
```

We also provided pre-trained models, and you can run the following code to test the model.

```
python test.py
```
## Citation
If you find our work is helpful, please cite as:


```
@article{wang2025hybrid,
  title={Hybrid spatial--temporal graph neural network for traffic forecasting},
  author={Wang, Peng and Feng, Longxi and Zhu, Yijie and Wu, Haopeng},
  journal={Information Fusion},
  pages={102978},
  year={2025},
  publisher={Elsevier}
}
```
