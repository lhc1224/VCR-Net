from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import shutil
import os
import cv2
import pdb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from lib.utils.tools.average_meter import AverageMeter
from lib.datasets.data_loader import DataLoader
from lib.loss.loss_manager import LossManager
from lib.models.model_manager import ModelManager
from lib.utils.tools.logger import Logger as Log
from lib.vis.seg_visualizer import SegVisualizer
from segmentor.tools.module_runner import ModuleRunner
from segmentor.tools.optim_scheduler import OptimScheduler
from segmentor.tools.data_helper import DataHelper
from segmentor.tools.evaluator import get_evaluator
from lib.utils.distributed import get_world_size, get_rank, is_distributed
from lib.utils.tools.configer import Configer
import argparse
from dataset.dataset_train import AFF_data,AFF_data_val
import torch.utils.data as data
import torchvision.utils as vutils
from Model.VCR_Net import EncoderDecoder
from lib.utils.evaluation import KLD,SIM,NSS,CC
import random


val_gt_path="dataset/Seen/gt_Seen_non_interactive_val.t7"
val_img_root="dataset/Seen/val/"
val_json="dataset/Seen/val.json"

class Trainer(object):
    
    def __init__(self, configer):
        self.configer = configer
        self.batch_time = AverageMeter()
        self.foward_time = AverageMeter()
        self.backward_time = AverageMeter()
        self.loss_time = AverageMeter()
        self.data_time = AverageMeter()
        self.train_losses = AverageMeter()
        self.val_losses = AverageMeter()
        self.seg_visualizer = SegVisualizer(configer)
        self.loss_manager = LossManager(configer)
        self.module_runner = ModuleRunner(configer)
        self.model_manager = ModelManager(configer)
        self.data_loader = DataLoader(configer)
        self.optim_scheduler = OptimScheduler(configer)
        self.data_helper = DataHelper(configer, self)
        self.evaluator = get_evaluator(configer, self)

        self.model = None
        self.train_loader = None
        self.val_loader = None
        self.optimizer = None
        self.scheduler = None
        self.running_score = None

        self._init_model()

    def _init_model(self):
        self.model= EncoderDecoder(backbone_name=self.configer.get("network","backbone"), ####  修改
                                  num_classes=7,
                                  pretrained_path=self.configer.get("network","pretrained"))
        Log.info(str(self.model))
       
        self.model = self.module_runner.load_net(self.model)
        num_params = 0
        for p in self.model.parameters():
            num_params += p.numel()
        Log.info("The number of parameters: {}".format(num_params))

        Log.info(
            "Params Group Method: {}".format(self.configer.get("optim", "group_method"))
        )
        params_group = self._get_parameters()

        Log.info(
            "Params Group Method: {}".format(self.configer.get("optim", "group_method"))
        )

        self.optimizer, self.scheduler = self.optim_scheduler.init_optimizer(
            params_group
        )
        self.best_kld=1000
        self.best_sim=-100
        self.best_nss=-100
        self.best_cc=-100
        self.gts=torch.load(val_gt_path)

        train_dst=AFF_data(image_root=self.configer.get("data","data_dir"),
                          crop_size=224)
        self.train_loader = data.DataLoader(train_dst, 
                                batch_size=self.configer.get("train","batch_size"),
                                shuffle=True, 
                                num_workers=4,pin_memory=False)
        val_dst=AFF_data_val(image_root=val_img_root,
                            val_json_path=val_json,
                            crop_size=224)
        self.val_loader = data.DataLoader(val_dst, batch_size=32,shuffle=False, 
                                         num_workers=4,pin_memory=False)

    @staticmethod
    def group_weight(module):
        group_decay = []
        group_no_decay = []
        for m in module.modules():
            if isinstance(m, nn.Linear):
                group_decay.append(m.weight)
                if m.bias is not None:
                    group_no_decay.append(m.bias)
            elif isinstance(m, nn.modules.conv._ConvNd):
                group_decay.append(m.weight)
                if m.bias is not None:
                    group_no_decay.append(m.bias)
            else:
                if hasattr(m, "weight"):
                    group_no_decay.append(m.weight)
                if hasattr(m, "bias"):
                    group_no_decay.append(m.bias)
                if hasattr(m, "absolute_pos_embed"):
                    group_no_decay.append(m.absolute_pos_embed)
                if hasattr(m, "relative_position_bias_table"):
                    group_no_decay.append(m.relative_position_bias_table)

        assert len(list(module.parameters())) == len(group_decay) + len(group_no_decay)
        groups = [
            dict(params=group_decay),
            dict(params=group_no_decay, weight_decay=0.0),
        ]
        return groups

    def _get_parameters(self):
        bb_lr = []
        nbb_lr = []
        params_dict = dict(self.model.named_parameters())

        for key, value in params_dict.items():
            if ("backbone" not in key):
                nbb_lr.append(value)
            else:
                bb_lr.append(value)
                Log.info(key)
            

        params = [
            {"params": bb_lr, "lr": self.configer.get("lr", "base_lr")},
            {
                "params": nbb_lr,
                "lr": self.configer.get("lr", "base_lr")
                * self.configer.get("lr", "nbb_mult"),
            },
        ]
        return params

    def __train(self):
        """
        Train function of every epoch during train phase.
        """
        scaler = torch.cuda.amp.GradScaler()
        self.model.train() 
        start_time = time.time()
        
        if hasattr(self.train_loader.sampler, "set_epoch"):
            self.train_loader.sampler.set_epoch(self.configer.get("epoch"))

        for i, dict_data in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            inputs, labels = dict_data['interactive_image'].cuda(), dict_data['interactive_target'].cuda()
            ego_images,ego_labels=dict_data['non_interactive_image'].cuda(),dict_data['non_interactive_target'].cuda()
            whole_body_pose=dict_data['human_dict']['whole_body_pose'].cuda()
            h_o_mask=dict_data['h_o_mask'].cuda()
            if self.configer.get("lr", "metric") == "iters":
                self.scheduler.step(self.configer.get("iters"))
            else:
                self.scheduler.step(self.configer.get("epoch"))

            if self.configer.get("lr", "is_warm"):
                self.module_runner.warm_lr(
                    self.configer.get("iters"),
                    self.scheduler,
                    self.optimizer,
                    backbone_list=[
                        0,
                    ],
                )
            self.data_time.update(time.time() - start_time)

            foward_start_time = time.time()
           
            out_1,out_2,_,cors_loss = self.model(inputs,ego_images,
                                                             whole_body_pose,h_o_mask)
            self.foward_time.update(time.time() - foward_start_time)

            loss_start_time = time.time()
            batch_size=labels.size(0)
           
            backward_loss = display_loss = F.binary_cross_entropy(out_1,labels,reduction='mean') \
                                         + F.binary_cross_entropy(out_2,ego_labels,reduction="mean") \
                                         + cors_loss 
            #pdb.set_trace()
            self.train_losses.update(display_loss.item(), batch_size)
            self.loss_time.update(time.time() - loss_start_time)

            backward_start_time = time.time()

            scaler.scale(backward_loss).backward()
            scaler.step(self.optimizer)
            scaler.update()
        
            
            self.backward_time.update(time.time() - backward_start_time)

            # Update the vars of the train phase.
            self.batch_time.update(time.time() - start_time)
            start_time = time.time()
            self.configer.plus_one("iters")

            # Print the log info & reset the states.
            if self.configer.get("iters") % self.configer.get(
                "solver", "display_iter") == 0:
                Log.info(
                    "Train Epoch: {0}\tTrain Iteration: {1}\t"
                    "Time {batch_time.sum:.3f}s / {2}iters, ({batch_time.avg:.3f})\t"
                    "Forward Time {foward_time.sum:.3f}s / {2}iters, ({foward_time.avg:.3f})\t"
                    "Backward Time {backward_time.sum:.3f}s / {2}iters, ({backward_time.avg:.3f})\t"
                    "Loss Time {loss_time.sum:.3f}s / {2}iters, ({loss_time.avg:.3f})\t"
                    "Data load {data_time.sum:.3f}s / {2}iters, ({data_time.avg:3f})\n"
                    "Learning rate = {3}\tLoss = {loss.val:.8f} (ave = {loss.avg:.8f})\n".format(
                        self.configer.get("epoch"),
                        self.configer.get("iters"),
                        self.configer.get("solver", "display_iter"),
                        self.module_runner.get_lr(self.optimizer),
                        batch_time=self.batch_time,
                        foward_time=self.foward_time,
                        backward_time=self.backward_time,
                        loss_time=self.loss_time,
                        data_time=self.data_time,
                        loss=self.train_losses,
                    )
                )
                self.batch_time.reset()
                self.foward_time.reset()
                self.backward_time.reset()
                self.loss_time.reset()
                self.data_time.reset()
                self.train_losses.reset()
            
            if not os.path.exists("tmp_see"):
                os.makedirs("tmp_see")
            if (i + 1) % 10 == 0:
                vutils.save_image(inputs.data[0,:,:,:],'tmp_see/iter%d-img.png'%i,padding=0)
                target=labels.view(-1,7,1,224,224)
                preds=out_1.view(-1,7,1,224,224)
                vutils.save_image(target.data[0,:,:,:,:],'tmp_see/iter%d-target.png'%i,padding=0)
                vutils.save_image(preds.data[0,:,:,:,:],'tmp_see/iter%d-preds.png'%i,padding=0)

            if self.configer.get("iters") == self.configer.get("solver", "max_iters"):
                break

        self.configer.plus_one("epoch")
        if (self.configer.get("epoch"))%3==0:
            file_root=self.configer.get("checkpoints","checkpoints_root")
            if not os.path.exists(file_root):
                os.makedirs(file_root,exist_ok=True)
            
            
            mkld,msim,mnss,mcc=self.eval_model()
            Log.info("mkld=%s,msim=%s,mnss=%s,mcc=%s" % (str(round(mkld,3)),str(round(msim,3)),str(round(mnss,3)),str(round(mcc,3))))
            if (mkld<self.best_kld) or (msim>self.best_sim) or  \
                (mnss>self.best_nss) or (mcc>self.best_cc): 
                torch.save(self.model.state_dict(), os.path.join(file_root, \
                                            "epoch_%s_best_%s_%s_%s_%s.pth" % (str(self.configer.get("epoch")), \
                                            str(round(mkld,3)),\
                                            str(round(msim,3)),str(round(mnss,3)),str(round(mcc,3)))))
                
                self.best_kld=min(mkld,self.best_kld)
                self.best_sim=max(msim,self.best_sim)
                self.best_nss=max(mnss,self.best_nss)
                self.best_cc=max(mcc,self.best_cc)
            if (self.configer.get("epoch")) % 20 == 0:
                torch.save(self.model.state_dict(), os.path.join(file_root, \
                                            "epoch_%s_%s_%s_%s_%s.pth" % (str(self.configer.get("epoch")), \
                                            str(round(mkld,3)),\
                                            str(round(msim,3)),str(round(mnss,3)),str(round(mcc,3)))))

    def eval_model(self):
        KLD_list=[]
        SIM_list=[]
        NSS_list=[]
        CC_list=[]
        self.model.eval()
        with torch.no_grad():
            for i, dict_data in enumerate(self.val_loader):
                inputs = dict_data['interactive_image'].cuda()
                ego_images=dict_data['non_interactive_image'].cuda()
                h_o_mask=dict_data['h_o_mask'].cuda()
                key=dict_data['key']
                whole_body_pose=dict_data['human_dict']['whole_body_pose'].cuda()
                out_1,out_2,_,_= self.model(inputs,ego_images,whole_body_pose,h_o_mask)
                #outputs = self.seg_net(x.cuda())
                outputs=out_2
                for idx in range(outputs.size(0)):
                    for j in range(outputs.size(1)):
                        pred_mask = outputs[idx,j,:,:].detach().cpu()
                        pred_mask = pred_mask.numpy()

                        gt_key="index_"+key[idx]+"_mask_"+str(j).zfill(3)
                        if gt_key in self.gts:
                            gt=self.gts[gt_key]
                            pred_mask=pred_mask/(np.max(pred_mask)+1e-12)
                            if gt.shape!=pred_mask.shape:
                                pred_mask=cv2.resize(pred_mask,(gt.shape[0],gt.shape[1]))
                            if np.max(gt)>0:
                                kld=KLD(pred_mask,gt)
                                sim=SIM(pred_mask,gt)
                                nss=NSS(pred_mask,gt)
                                cc=CC(pred_mask,gt)
                                KLD_list.append(kld)
                                SIM_list.append(sim)
                                if not np.isnan(cc):
                                    CC_list.append(cc)
                                if not np.isnan(nss):
                                    NSS_list.append(nss)
        mkld=sum(KLD_list)/(len(KLD_list))
        msim=sum(SIM_list)/(len(SIM_list))
        mnss=sum(NSS_list)/(len(NSS_list))
        mcc=sum(CC_list)/(len(CC_list))
        self.model.train()
        #print("mkid=%s"%str(mkld))
        return [mkld,msim,mnss,mcc]

    def train(self):
    
        while self.configer.get("iters") < self.configer.get("solver", "max_iters"):
            self.__train()

        # use swa to average the model
        if "swa" in self.configer.get("lr", "lr_policy"):
            self.optimizer.swap_swa_sgd()
            self.optimizer.bn_update(self.train_loader, self.model)

def str2bool(v):
    """Usage:
    parser.add_argument('--pretrained', type=str2bool, nargs='?', const=True,
                        dest='pretrained', help='Whether to use pretrained models.'
                        )
    """
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Unsupported value encountered.")

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--configs",
        default="configs/coco_stuff/H_48_D_4.json",
        type=str,
        dest="configs",
        help="The file of the hyper parameters.",
    )
    parser.add_argument(
        "--phase", default="train", type=str, dest="phase", help="The phase of module."
    )
    parser.add_argument(
        "--gpu",
        default="0",
        nargs="+",
        type=str,
        dest="gpu",
        help="The gpu list used.",
    )
    
    # ***********  Params for data.  **********
    parser.add_argument(
        "--data_dir",
        default="dataset/Seen/train/interactive/",
        type=str,
        nargs="+",
        dest="data:data_dir",
        help="The Directory of the data.",
    )
    
    parser.add_argument(
        "--drop_last",
        type=str2bool,
        nargs="?",
        default=False,
        dest="data:drop_last",
        help="Fix bug for syncbn.",
    )
    parser.add_argument(
        "--workers",
        default=4,
        type=int,
        dest="data:workers",
        help="The number of workers to load data.",
    )
    parser.add_argument(
        "--train_batch_size",
        default=24,
        type=int,
        dest="train:batch_size",
        help="The batch size of training.",
    )
    # ***********  Params for checkpoint.  **********
    parser.add_argument(
        "--checkpoints_root",
        default="save_models/",
        type=str,
        dest="checkpoints:checkpoints_root",
        help="The root dir of model save path.",
    )
    parser.add_argument(
        "--checkpoints_name",
        default=None,
        type=str,
        dest="checkpoints:checkpoints_name",
        help="The name of checkpoint model.",
    )
    parser.add_argument(
        "--save_epoch",
        default=5,
        type=int,
        dest="checkpoints:save_epoch",
        help="The saving epoch of checkpoint model.",
    )
    # ***********  Params for model.  **********
    parser.add_argument(
        "--model_name",
        default="segformer",
        type=str,
        dest="network:model_name",
        help="The name of model.",
    )
    parser.add_argument(
        "--backbone",
        default="mit_b2",
        type=str,
        dest="network:backbone",
        help="The base network of model.",
    )
    parser.add_argument(
        "--bn_type",
        default="torchbn",
        type=str,
        dest="network:bn_type",
        help="The BN type of the network.",
    )
    parser.add_argument(
        "--multi_grid",
        default=None,
        nargs="+",
        type=int,
        dest="network:multi_grid",
        help="The multi_grid for resnet backbone.",
    )
    parser.add_argument(
        "--pretrained",
        type=str,
        default="pretrained/mit_b2.pth",
        dest="network:pretrained",
        help="The path to pretrained model.",
    )
    parser.add_argument(
        "--resume",
        default=None,
        type=str,
        dest="network:resume",
        help="The path of checkpoints.",
    )
    parser.add_argument(
        "--resume_strict",
        type=str2bool,
        nargs="?",
        default=True,
        dest="network:resume_strict",
        help="Fully match keys or not.",
    )
    parser.add_argument(
        "--resume_continue",
        type=str2bool,
        nargs="?",
        default=False,
        dest="network:resume_continue",
        help="Whether to continue training.",
    )
    parser.add_argument(
        "--resume_eval_train",
        type=str2bool,
        nargs="?",
        default=True,
        dest="network:resume_train",
        help="Whether to validate the training set  during resume.",
    )
    parser.add_argument(
        "--resume_eval_val",
        type=str2bool,
        nargs="?",
        default=True,
        dest="network:resume_val",
        help="Whether to validate the val set during resume.",
    )
    parser.add_argument(
        "--gathered",
        type=str2bool,
        nargs="?",
        default=False,
        dest="network:gathered",
        help="Whether to gather the output of model.",
    )
   

    # ***********  Params for solver.  **********
    parser.add_argument(
        "--optim_method",
        default="adamw",
        type=str,
        dest="optim:optim_method",
        help="The optim method that used.",
    )
    parser.add_argument(
        "--group_method",
        default="decay",
        type=str,
        dest="optim:group_method",
        help="The group method that used.",
    )
    parser.add_argument(
        "--base_lr",
        default=6e-5,
        type=float,
        dest="lr:base_lr",
        help="The learning rate.",
    )
    parser.add_argument(
        "--nbb_mult",
        default=10.0,
        type=float,
        dest="lr:nbb_mult",
        help="The not backbone mult ratio of learning rate.",
    )
    parser.add_argument(
        "--lr_policy",
        default="warm_lambda_poly",
        type=str,
        dest="lr:lr_policy",
        help="The policy of lr during training.",
    )
    
    parser.add_argument(
        "--is_warm",
        type=str2bool,
        nargs="?",
        default=False,
        dest="lr:is_warm",
        help="Whether to warm training.",
    )
    # ***********  Params for display.  **********
    parser.add_argument(
        "--max_epoch",
        default=200,
        type=int,
        dest="solver:max_epoch",
        help="The max epoch of training.",
    )
    parser.add_argument(
        "--max_iters",
        default=180000,
        type=int,
        dest="solver:max_iters",
        help="The max iters of training.",
    )
    parser.add_argument(
        "--display_iter",
        default=50,
        type=int,
        dest="solver:display_iter",
        help="The display iteration of train logs.",
    )
    # ***********  Params for logging.  **********
    parser.add_argument(
        "--logfile_level",
        default="info",
        type=str,
        dest="logging:logfile_level",
        help="To set the log level to files.",
    )
    parser.add_argument(
        "--stdout_level",
        default="info",
        type=str,
        dest="logging:stdout_level",
        help="To set the level to print to screen.",
    )
    parser.add_argument(
        "--log_file",
        default="log/segformer.log",
        type=str,
        dest="logging:log_file",
        help="The path of log files.",
    )
    parser.add_argument(
        "--rewrite",
        type=str2bool,
        nargs="?",
        default=True,
        dest="logging:rewrite",
        help="Whether to rewrite files.",
    )
    parser.add_argument(
        "--log_to_file",
        type=str2bool,
        nargs="?",
        default=True,
        dest="logging:log_to_file",
        help="Whether to write logging into files.",
    )
    # ***********  Params for env.  **********
    parser.add_argument("--seed", default=0, type=int, help="manual seed")
    parser.add_argument(
        "--cudnn", type=str2bool, nargs="?", default=True, help="Use CUDNN."
    )

    parser.add_argument("REMAIN", nargs="*")
    args_parser = parser.parse_args()
    time_str=time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))
    

    configer = Configer(args_parser=args_parser)
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args_parser.gpu[0]
    configer.update(["checkpoints","checkpoints_root"], os.path.join(configer.get("checkpoints","checkpoints_root"),time_str))
    if configer.get("logging", "log_to_file"):
        log_file = configer.get("logging", "log_file")

        new_log_file=os.path.join(configer.get("checkpoints","checkpoints_root"),
                                  log_file)
        print(new_log_file)
        configer.update(["logging", "log_file"], new_log_file)
    else:
        configer.update(["logging", "logfile_level"], None)
  
    Log.init(
        logfile_level=configer.get("logging", "logfile_level"),
        stdout_level=configer.get("logging", "stdout_level"),
        log_file=configer.get("logging", "log_file"),
        log_format=configer.get("logging", "log_format"),
        rewrite=configer.get("logging", "rewrite"),
    )

    setup_seed(configer.get('seed'))
    shutil.copy(
        'train.py',
        configer.get("checkpoints","checkpoints_root"))
    shutil.copy(
    'Model/VCR_Net.py',
    configer.get("checkpoints","checkpoints_root"))

    model = Trainer(configer)
    
    model.train()

    



