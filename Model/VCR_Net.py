from curses.ascii import CR
from operator import indexOf
import torch
import torch.nn as nn
import torch.nn.functional as F

#from mmseg.core import add_prefix
from mmseg.ops import resize
from timm.models.vision_transformer import vit_base_patch16_224
from Model.mix_transformer import mit_b0,mit_b1, mit_b2, mit_b3,mit_b4,mit_b5
from Model.decoder import SegFormerHead
from Model.simple_vit_1d import SimpleViT
#from Model.DEQ_Fuse import DEQFusion
from Model.DEQ_Sequence.models.deq_fuse_layer import DEQTransformerLM
from Model.lib.solvers import anderson, broyden
from Model.text import bert_embed,tokenize,BERT_MODEL_DIM
from Model.cross_transformer import CrossTransformer
import pdb
import torch.nn.functional as F


class Conv_BN_RELU(nn.Module):
    def __init__(self,in_ch,out_ch,kernel_size=1,stride=1,padding=0):
        super(Conv_BN_RELU, self).__init__()
        self.layer=nn.Sequential(nn.Conv2d(in_channels=in_ch,out_channels=out_ch,
                                    kernel_size=kernel_size,
                                    stride=stride,padding=padding),
                                    nn.BatchNorm2d(out_ch),
                                    nn.ReLU())
    def forward(self,input):
        return self.layer(input)

class EncoderDecoder(nn.Module):
    """Encoder Decoder segmentors.

    EncoderDecoder typically consists of backbone, decode_head, auxiliary_head.
    Note that auxiliary_head is only used for deep supervision during training,
    which could be dumped during inference.
    """

    def __init__(self,
                 backbone_name,
                 num_classes=6,
                 pretrained_path=None):
        
        super(EncoderDecoder, self).__init__()
        

        self.backbone_name=backbone_name
        self.num_classes=num_classes
        self.pretrained_path=pretrained_path
        
        #### extract_feature
        self.reduce_dim=256
        self.init_backbone()  
        self.init_text()
        
        #self.text_enconde=bert_embed()  
        
        self.reduce_conv = nn.Sequential(nn.Conv2d(in_channels=512,out_channels=self.embed_dim,
                                        kernel_size=1,stride=1,padding=0),
                                        nn.BatchNorm2d(self.embed_dim),
                                        nn.ReLU())
        
        self.human_encoder = SimpleViT(dim = self.embed_dim,
                                     depth = 6,heads = 4,
                                     mlp_dim = self.embed_dim*4)
        
        self.fuse_body_interactive = DEQTransformerLM(n_layer=3, eval_n_layer=3, n_head=4, d_model=self.embed_dim, 
                                                    d_head=self.embed_dim//4, d_inner=self.embed_dim*4,
                                                    dropout=0.0, dropatt=0.0, d_embed=None, 
                                                    pre_lnorm=False, wnorm=False, tgt_len=49+53+1,
                                                    mem_len=49+53+1, local_size=0, pretrain_steps=0, 
                                                    f_solver=anderson, b_solver=anderson, stop_mode="abs", 
                                                    logging=None,idx_list=[49,49+53,49+53+1])
        
        self.fuse_conv = Conv_BN_RELU(in_ch = self.embed_dim, out_ch = self.embed_dim)
        
        self.fuse_body_non_interactive = DEQTransformerLM(n_layer = 3, eval_n_layer = 3, n_head = 4, d_model = self.embed_dim, 
                                                    d_head = self.embed_dim//4, d_inner = self.embed_dim*4,
                                                    dropout = 0.0, dropatt=0.0, d_embed=None, 
                                                    pre_lnorm=False, wnorm=False, tgt_len=49+53,
                                                    mem_len=49+53, local_size=0, pretrain_steps=0, 
                                                    f_solver=anderson, b_solver=anderson, stop_mode="abs", 
                                                    logging=None,idx_list=[49,49+53])
        
        self.text_interactive_per = CrossTransformer(dim=self.embed_dim,
                                                    hidden_dim=self.embed_dim*4,
                                                    num_query_token=16, 
                                                    depth=1, heads=4, 
                                                    dim_head=self.embed_dim//4, dropout=0.1)
        self.reduce_text_feature=nn.Linear(in_features=BERT_MODEL_DIM,out_features=self.embed_dim)
        
        self.fuse_conv_2=Conv_BN_RELU(in_ch=self.embed_dim,out_ch=self.embed_dim)
        self.fuse_pose_text_fea=Conv_BN_RELU(in_ch=self.embed_dim*2,out_ch=self.embed_dim)
        
        self.app_fuse=DEQTransformerLM(n_layer=3, eval_n_layer=3, n_head=4, d_model=256, 
                                        d_head=64, d_inner=1024,
                                        dropout=0.0, dropatt=0.0, d_embed=None, 
                                        pre_lnorm=False, wnorm=False, tgt_len=49+53+self.num_classes,
                                        mem_len=49+53+self.num_classes, local_size=0, pretrain_steps=0, 
                                        f_solver=anderson, b_solver=anderson, stop_mode="abs", 
                                        logging=None,idx_list=[49,49+53,49+53+self.num_classes])
        self.fuse_conv_3=Conv_BN_RELU(in_ch=self.embed_dim*(self.num_classes+1),
                                      out_ch=self.embed_dim)

        self.fuse_head_1=SegFormerHead(feature_strides=[4, 8, 16, 32],
                                         in_channels=self.in_channels,
                                         dropout_ratio=0.1,
                                         embed_dim=self.embed_dim,
                                         in_index=[0, 1, 2, 3])
        
        self.fuse_head_2=SegFormerHead(feature_strides=[4, 8, 16, 32],
                                         in_channels=self.in_channels,
                                         dropout_ratio=0.1,
                                         embed_dim=self.embed_dim,
                                         in_index=[0, 1, 2, 3])
       
        self.align_corners = False
        
        self.linear_pred = nn.Sequential(nn.Conv2d(self.embed_dim, self.num_classes, kernel_size=1),
                                        nn.Sigmoid())
        self.linear_pred_2 = nn.Sequential(nn.Conv2d(self.embed_dim, self.num_classes, 
                                                   kernel_size=1),
                                        nn.Sigmoid())

        
        self.pool=nn.AdaptiveAvgPool2d(1)
    def init_text(self,):
        self.text_list=["This is the region where the hand interacts with the object.",
                        "This is the region where the feet interact with the object.",
                        "This is the region where the mouth interacts with the object.",
                        "This is the region where the mouth interacts with the object.",
                        "This is the region where the back interacts with the object.",
                        "This is the region where the hip interacts with the object.",
                        "This is the region where the eye interacts with the object.",
                        "This is the region where the object interacts with the outside environment."
                        ]
        self.text_feature_list=[]
        

    def init_backbone(self,):
        
        if self.backbone_name=="mit_b2":
            self.backbone=mit_b2()
            self.in_channels=[64, 128, 320, self.reduce_dim]
            self.embed_dim=256
           
        elif self.backbone_name=="mit_b3":
            self.backbone=mit_b3()
            self.in_channels=[64, 128, 320, self.reduce_dim]
            self.embed_dim=256
            
        else:
            self.backbone=mit_b5() 
            self.in_channels=[64, 128, 320, self.reduce_dim]
            self.embed_dim=512
        if self.pretrained_path!=None:
            self.backbone.init_weights(pretrained=self.pretrained_path)
    #### SHP module
    def affienty_extract(self,x_1,whole_body_feature,h_o_x_1):  
        b=x_1.size(0)
        w=x_1.size()[2]
        h=x_1.size()[3]
        if len(self.text_feature_list)==0:
            for text_input in self.text_list:
                text_feature= bert_embed(tokenize(text_input), return_cls_repr=False).cuda()
                self.text_feature_list.append(text_feature)

        ### text feature extract   
        self.text_features=self.reduce_text_feature(torch.stack(self.text_feature_list,dim=1))
        self.text_features=self.text_features.repeat(b,1,1).permute(0,2,1) 
        
        ### text feature fuse
        x_1_fuse_text=self.text_interactive_per(x_1.view(b,self.embed_dim,-1),
                                                  self.text_features)
        x_1_fuse_text=x_1_fuse_text.view(b,self.embed_dim,w,h)
        
        #### body feature fuse
        whole_body_feature=whole_body_feature.permute(0,2,1) ## [b,256,53]
        x_1_reshape=x_1.view(b,self.reduce_dim,-1)  ### [b,256,49]
        hidden_1,new_mems_1,jac_loss_1=\
                       self.fuse_body_interactive(torch.cat((x_1_reshape,whole_body_feature,h_o_x_1.unsqueeze(2)),dim=2)) ####
       
        hidden_1=hidden_1.permute(1,2,0).contiguous()
        fuse_x_1,whole_body_feat_1=hidden_1[:,:,:49],hidden_1[:,:,49:-1]
        
        fuse_x_1=self.fuse_conv(fuse_x_1.view(b,-1,7,7).contiguous() + x_1) ### 

        fuse_x_1=self.fuse_pose_text_fea(torch.cat((fuse_x_1,x_1_fuse_text),dim=1))

        return fuse_x_1,whole_body_feat_1,jac_loss_1[0][0]
    #### GAT module
    def pose_guide_align(self,whole_body_feat_1,x_2,whole_body_feature):
        b=x_2.size(0)
        w=x_2.size()[2]
        h=x_2.size()[3]
        x_2_reshape=x_2.view(b,self.reduce_dim,-1)  ### [b,256,49]
        #### pose enhance
        whole_body_feature=whole_body_feature.permute(0,2,1) ## [b,256,53]
        hidden_2,new_mems_2,jac_loss_2=\
                                   self.fuse_body_non_interactive(torch.cat((x_2_reshape,
                                                                             whole_body_feature),
                                                                             dim=2)) ###
        hidden_2=hidden_2.permute(1,2,0).contiguous()
        fuse_x_2,whole_body_feat_2=hidden_2[:,:,:49],hidden_2[:,:,49:]

        ##### kl align
        cors_loss=F.kl_div(whole_body_feat_1.softmax(dim=-1).log(), whole_body_feat_2.softmax(dim=-1), reduction='sum')
        fuse_x_2=self.fuse_conv_2(fuse_x_2.view(b,-1,7,7).contiguous() + x_2)
        return fuse_x_2,jac_loss_2[0][0],cors_loss,whole_body_feat_2
    
    def app_align(self,out_1,en_x_1,en_x_2,fuse_x_2,whole_pose_feat):
        b=out_1.size(0)
        en_x_1_list=[]  
        ex_x_1_list=[]
        for i in range(out_1.size(1)):
            
            mask_out=out_1[:,i,:,:].view(b,1,en_x_1.size(2),en_x_2.size(3)).repeat(1,self.embed_dim,1,1)
            tmp_out=self.pool(en_x_1*mask_out)  #### part rep
            
            tmp_out=tmp_out.view(b,self.embed_dim,1).contiguous()
            en_x_1_list.append(tmp_out)
            ex_x_1=tmp_out.view(b,-1,1,1).repeat(1,1,en_x_2.size(2),en_x_2.size(3))
            ex_x_1_list.append(ex_x_1)
        in_1_branch=torch.cat(en_x_1_list,dim=2)

        fuse_x_2_tmp=fuse_x_2.view(b,-1,7*7)
        hidden,new_mems,jac_loss=self.app_fuse(torch.cat((fuse_x_2_tmp,whole_pose_feat,in_1_branch),dim=2))

        hidden=hidden.permute(1,2,0).contiguous()
        fuse_x_2_update=hidden[:,:,:49].view(b,-1,7,7)
        fuse_x_2_update=resize(fuse_x_2_update,size=en_x_2.shape[2:],
                               mode='bilinear',
                               align_corners=self.align_corners)
        #pdb.set_trace()
        ex_x_1_feature=torch.stack(ex_x_1_list,dim=1)
        ex_x_1_feature=ex_x_1_feature.view(b,-1,en_x_2.size(2),en_x_2.size(3))
        fuse_x_2_update_fuse=self.fuse_conv_3(torch.cat((fuse_x_2_update+en_x_2,ex_x_1_feature),dim=1))
        out_2=self.linear_pred_2(fuse_x_2_update_fuse)
        return out_2,jac_loss[0][0]
        
    def forward(self,img,img_2,whole_body,h_o_mask):
        
        #### feature enconde 
        x_1=self.backbone(img)
        x_2=self.backbone(img_2)
        x_1_layer1,x_1_layer_2,x_1_layer_3,x_1_layer_4 = x_1
        x_2_layer1,x_2_layer_2,x_2_layer_3,x_2_layer_4 = x_2
        b,c,h,w = x_2_layer_4.size()

        whole_body_feature = self.human_encoder(whole_body)  ####  [b,53,256]   
        
        x_2_layer_4 = self.reduce_conv(x_2_layer_4)
        
        x_1_layer_4 = self.reduce_conv(x_1_layer_4)
        #pdb.set_trace()
        h_o_mask = F.interpolate(h_o_mask.unsqueeze(1),size=[7,7],mode="bilinear",align_corners=False)
        #pdb.set_trace()
        h_o_x_1=x_1_layer_4*h_o_mask
        area = F.avg_pool2d(h_o_mask, (w,h)) * h * w + 0.0005
        h_o_feat=F.adaptive_avg_pool2d(h_o_x_1,1)*w*h/area
        
        fuse_x_1,whole_body_feat_1,jac_loss_1=self.affienty_extract(x_1_layer_4,whole_body_feature,h_o_feat.squeeze(3).squeeze(2))
        
        en_x_1=self.fuse_head_1([x_1_layer1,x_1_layer_2,x_1_layer_3,fuse_x_1])
        out_1=self.linear_pred(en_x_1)  ### predicted 
        
        fuse_x_2,jac_loss_2,cors_loss,whole_body_feat_2 = self.pose_guide_align(whole_body_feat_1,x_2_layer_4,whole_body_feature)
        en_x_2=self.fuse_head_2([x_2_layer1,x_2_layer_2,x_2_layer_3,fuse_x_2])

        jacobian_loss=jac_loss_1+jac_loss_2

        
        out_2,jac_loss=self.app_align(out_1,en_x_1,en_x_2,fuse_x_2,whole_body_feat_2)
        jacobian_loss+=jac_loss
       
        out_2 = resize(
            input=out_2,
            size=img_2.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        out_1 = resize(
            input=out_1,
            size=img.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        
        return out_1,out_2,jacobian_loss,cors_loss

   
