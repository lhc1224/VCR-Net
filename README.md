# VCR-Net
PyTorch implementation of our VCR-Net models. 

4. [Requirements](#1)
5. [Download CAL dataset](#2)
6. [Test](#3)
7. [Citation](#4)



## Requirements <a name="1"></a>
Environment installation as follows [Segformer](https://github.com/NVlabs/SegFormer)
- python 3.9
```bash 
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html
pip install scipy
pip install einops
pip install torchcontrib
pip install ipython
pip install attr
pip install termcolor
pip install transformers
pip install scikit-image
pip install scikit-learn
```

### Download CAL dataset <a name="2"></a> 
- You can download the CAL dataset from [ [Onedrive]() | [Baidu Pan](https://pan.baidu.com/s/1D5wAZfVWcY94ijYki3z7Fw)(q9bh)  ].
Download the dataset and place it in the dataset/ folder

### Test <a name="3"></a> 
You can test the trained model by running `test.py`.

Download the pretrained models from [ [Onedrive](https://1drv.ms/f/s!AvZMjCSI4SWDbk7Rqn5I2apUhL8?e=Lrs3qR) | [Baidu Pan](https://pan.baidu.com/s/1sEy8eJZbmcjEGAteU3JWaw)(7itq)  ] and place it in the pretrained/ folder

Download the Bert model from [ [Onedrive](https://1drv.ms/f/s!AvZMjCSI4SWDcDIlzpxgSv6A2ao?e=3FaNcw) | [Baidu Pan](https://pan.baidu.com/s/1_VvmPdn7BmQD-nIydsL7rw)(p468)  ] and place it in the bert-base-cased/ folder

Download the models from [ [Onedrive](https://1drv.ms/f/s!AvZMjCSI4SWDbFBLTJIy8UfL4q8?e=gf3XSi) | [Baidu Pan](https://pan.baidu.com/s/1J7NHNZ5mBEAGoiW3yf7qLg)(2sd3)  ] and place it in the save_models/ folder

```bash  
 python test.py  
```
## Citation <a name="4"></a> 

```

```

