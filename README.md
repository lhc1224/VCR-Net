# VCR-Net
PyTorch implementation of our VCR-Net models. 

1. [Paper Link](#1)
2. [Abstract](#2)
4. [Requirements](#3)
5. [Download CAL dataset](#4)
6. [Test](#5)
7. [Citation](#6)

## 📎 Paper Link <a name="1"></a> 
* Visual-geometric Collaborative Guidance for Affordance Learning  [[pdf]()] 
> Authors:
> Hongchen Luo, Wei Zhai, Jiao Wang, Yang Cao, Zheng-Jun Zha


## 💡 Abstract <a name="2"></a> 
Perceiving potential "action possibilities" (i.e., affordance) regions of images and learning interactive functionalities of objects from human demonstration is a challenging task due to the diversity of human-object interactions. Prevailing affordance learning algorithms often adopt the label assignment paradigm and presume that there is a unique relationship between functional region and affordance label, yielding poor performance when adapting to unseen environments with large appearance variations. In this paper, we propose to leverage interactive affinity for affordance learning, i.e., extracting interactive affinity from human-object interaction and transferring it to non-interactive objects. Interactive affinity, which represents the contacts between different parts of the human body and local regions of the target object, can provide inherent cues of interconnectivity between humans and objects, thereby reducing the ambiguity of the perceived action possibilities. To this end, we propose a visual-geometric collaborative guided affordance learning network that incorporates visual and geometric cues to excavate interactive affinity from human-object interactions jointly. Particularly, a semantic-pose heuristic perception (SHP) module is devised to exploit both semantic and geometric cues to guide the network to focus on interaction-relevant regions, alleviating the effects of combinatorial relational ambiguity. Meanwhile, A geometric-apparent alignment transfer module is introduced to jointly align local regions of apparent and structural similarity, eliminating the transport difficulties posed by intra-class correspondence ambiguity. Besides, a contact-driven affordance learning (CAL) dataset is constructed by collecting and labeling over 55,047 images. Experimental results demonstrate that our method outperforms the representative models regarding objective metrics and visual quality. 



## Requirements <a name="3"></a>
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

### Download CAL dataset <a name="4"></a> 
- You can download the CAL dataset from [ [Onedrive]() | [Baidu Pan](https://pan.baidu.com/s/1D5wAZfVWcY94ijYki3z7Fw)(q9bh)  ].
Download the dataset and place it in the dataset/ folder

### Test <a name="5"></a> 
You can test the trained model by running `test.py`.

Download the pretrained models from [ [Onedrive](https://1drv.ms/f/s!AvZMjCSI4SWDbk7Rqn5I2apUhL8?e=Lrs3qR) | [Baidu Pan](https://pan.baidu.com/s/1sEy8eJZbmcjEGAteU3JWaw)(7itq)  ] and place it in the pretrained/ folder

Download the Bert model from [ [Onedrive](https://1drv.ms/f/s!AvZMjCSI4SWDcDIlzpxgSv6A2ao?e=3FaNcw) | [Baidu Pan](https://pan.baidu.com/s/1_VvmPdn7BmQD-nIydsL7rw)(p468)  ] and place it in the bert-base-cased/ folder

Download the models from [ [Onedrive](https://1drv.ms/f/s!AvZMjCSI4SWDbFBLTJIy8UfL4q8?e=gf3XSi) | [Baidu Pan](https://pan.baidu.com/s/1J7NHNZ5mBEAGoiW3yf7qLg)(2sd3)  ] and place it in the save_models/ folder

```bash  
 python test.py  
```
## Citation <a name="6"></a> 

```

```

