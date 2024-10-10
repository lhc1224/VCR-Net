from email.mime import image
import json
import os
from collections import namedtuple
from tkinter.messagebox import NO
from torchvision import transforms
import cv2
import numpy as np
import torch.utils.data as data
from PIL import Image
import torch
import json
import os
from collections import namedtuple
import torchvision
import torch.utils.data as data
from PIL import Image
from random import choice
import glob


mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

class AFF_data(data.Dataset):

    def __init__(self,image_root,transform=None,crop_size=224):
        self.interactive_images = []
        self.non_interactive_images=[]
        self.images=[]
        self.targets=[]
        self.mesh_path=[]
        
        self.crop_size=crop_size
        self.interactive_root=image_root
        self.non_interactive_root=self.interactive_root.replace("interactive","non_interactive")
        self.mesh_root=self.interactive_root.replace("interactive","interactive_save_mesh")
        self.label_root=self.interactive_root.replace("interactive","interactive_label")

        files=os.listdir(image_root)
        for file in files:
            file_path=os.path.join(image_root,file)
            sub_files=os.listdir(file_path)
            for sub_file in sub_files:
                sub_path=os.path.join(file_path,sub_file)

                images=os.listdir(sub_path)
                for img in images:
                    if img[-4:]=="json":
                        continue
                    img_path=os.path.join(sub_path,img)
                    
                    label_path=os.path.join(self.label_root,file,sub_file,img[:-3]+"json")
                    mesh_path=os.path.join(self.mesh_root,file,sub_file,img[:-4],"smplx")
                    if os.path.exists(label_path) and os.path.exists(mesh_path):
                        self.images.append(img_path)
                        self.interactive_images.append(img_path)
                                
        files=os.listdir(self.non_interactive_root)
        for file in files:
            file_path=os.path.join(self.interactive_root,file)
            sub_files=os.listdir(file_path)
            for sub_file in sub_files:
                sub_path=os.path.join(file_path,sub_file)

                images=os.listdir(sub_path)
                for img in images:
                    if img[-4:]=="json":
                        continue
                    img_path=os.path.join(sub_path,img)
                    self.images.append(img_path)
                    self.non_interactive_images.append(img_path)
                        
        self.transform = transforms.Compose([transforms.Resize((crop_size,crop_size)),
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean, std)])
        
    def read_json(self,json_path,npz_path=None,interactive_flag=False):
        dict_1 = {}
        dict_1["point1"] = []
        dict_1["point2"] = []
        dict_1['point3'] = []
        dict_1['point4'] = []
        dict_1['point5'] = []
        dict_1['point6'] = []
        dict_1['point7'] = []
        
        with open(json_path, 'r') as load_f:
            json_data = json.load(load_f)
            dict_1['image_height'] = json_data['imageHeight']
            dict_1['image_width'] = json_data["imageWidth"]
            instances=json_data['instance']
            num_instance=len(instances)
            if interactive_flag==True:
                if num_instance>1:
                    idx=np.random.randint(0,num_instance)
                else:
                    idx=0
                if os.path.exists(os.path.join(npz_path,str(idx+1)+".npz")):
                    human_dict=load_smplx_data(os.path.join(npz_path,str(idx+1)+".npz"))
                else:
                    human_dict={}
                    human_dict['global_orient']=torch.zeros(1,3)
                    human_dict['body_pose']=torch.zeros(21,3)
                    human_dict['left_hand_pose']=torch.zeros(15,3)
                    human_dict['right_hand_pose']=torch.zeros(15,3)
                    human_dict['jaw_pose']=torch.zeros(1,3)
                    human_dict['whole_body_pose']=torch.zeros(53,3)
                shapes=instances[idx]
                box_list=[]
                for shape in shapes:
                    if shape['shape_type'] == "point":
                        dict_1[shape['label']].append(shape['points'][0])
                    if shape["shape_type"]=="rectangle":
                        box_list.append(shape["points"])
                    
            else:
                for shapes in instances:
                    for shape in shapes:
                        if shape['shape_type'] == "point":
                            dict_1[shape['label']].append(shape['points'][0])
        load_f.close()

        heatmaps = gen_heatmap(dict_1,flag="train",crop_size=self.crop_size)
        if interactive_flag==True:
            if len(box_list)==2:
                h_o_x1=int(max(min(box_list[0][0][0],box_list[1][0][0])-20,0))
                h_o_x2=int(min(max(box_list[0][1][0],box_list[1][1][0])+20, dict_1['image_height'])) ####
                h_o_y1=int(max(min(box_list[0][0][1],box_list[1][0][1])-20,0))
                h_o_y2=int(min(max(box_list[0][1][1],box_list[1][1][1])+20, dict_1['image_width']))
                mask = np.zeros((dict_1['image_height'], dict_1['image_width']), dtype=np.float32)
                mask[h_o_y1:h_o_y2,h_o_x1:h_o_x2]=1
                mask = cv2.resize(mask, (self.crop_size, self.crop_size))
                mask=torch.from_numpy(mask)
            else:
                mask=torch.ones((self.crop_size,self.crop_size))
            
            return heatmaps,mask,human_dict
        else:
            return heatmaps

    def __getitem__(self, index):

        img_path=self.images[index]
        dict_data={}
        if img_path in self.interactive_images:
            image = Image.open(img_path).convert('RGB')
            aff,obj,img_name=img_path.split("/")[-3],img_path.split("/")[-2],img_path.split("/")[-1]
            label_path=os.path.join(self.label_root,aff,obj,img_name[:-3]+"json")
            npz_path=os.path.join(self.mesh_root,aff,obj,img_name[:-4],"smplx")
            
            interactive_target,h_o_mask,human_dict=self.read_json(label_path,npz_path,interactive_flag=True)
            interactive_image=self.transform(image)

            non_interactive_img_list=glob.glob(os.path.join(self.non_interactive_root,aff,"*/*jpg"))
            
            non_interactive_sample_file=choice(non_interactive_img_list)
            non_interactive_image=Image.open(non_interactive_sample_file).convert('RGB')
            non_interactive_target_path=non_interactive_sample_file.replace("jpg","json")
            non_interactive_target=self.read_json(non_interactive_target_path)
            non_interactive_image=self.transform(non_interactive_image)

        else:
            non_interactive_image=Image.open(img_path).convert("RGB")
            non_interactive_target_path=img_path.replace("jpg","json")
            non_interactive_target=self.read_json(non_interactive_target_path)
            non_interactive_image=self.transform(non_interactive_image)
            aff=img_path.split("/")[-3]
            
            interactive_img_list=glob.glob(os.path.join(self.interactive_root,aff,"*/*jpg"))

            interactive_sample_file=choice(interactive_img_list)
            obj,img_name=interactive_sample_file.split("/")[-2],interactive_sample_file.split("/")[-1]
            
            
            label_path=os.path.join(self.label_root,aff,obj,img_name[:-3]+'json')
            while not os.path.exists(os.path.join(self.mesh_root,aff,obj,img_name[:-4],"smplx")):
                interactive_sample_file=choice(interactive_img_list)
                obj,img_name=interactive_sample_file.split("/")[-2],interactive_sample_file.split("/")[-1]

            npz_path=os.path.join(self.mesh_root,aff,obj,img_name[:-4],"smplx")

            image = Image.open(interactive_sample_file).convert('RGB')
            interactive_target,h_o_mask,human_dict=self.read_json(label_path,npz_path,interactive_flag=True)
            interactive_image=self.transform(image)

        dict_data['interactive_image']=interactive_image
        dict_data['interactive_target']=torch.stack(interactive_target,dim=0)
        dict_data['h_o_mask']=h_o_mask
        dict_data['human_dict']=human_dict
        dict_data['non_interactive_image']=non_interactive_image
        dict_data['non_interactive_target']=torch.stack(non_interactive_target,dim=0)
        return dict_data

    def __len__(self):
        return len(self.images)

class AFF_data_val(data.Dataset):

    def __init__(self,image_root,
                 val_json_path,
                 crop_size=224):

        self.interactive_images=[]
        self.non_interactive_images=[]
        self.image_root=image_root
        self.crop_size=crop_size
        
        self.mesh_root=os.path.join(self.image_root,"interactive_save_mesh")
        self.label_root=os.path.join(self.image_root,"interactive_label")
        self.targets = []
        self.keys=[]
        self.transform = transforms.Compose([transforms.Resize((crop_size,crop_size)),
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean, std)])
        self.image_pairs=[]
        
        with open(val_json_path, 'r') as load_f:
            json_data = json.load(load_f)
            for key,value in json_data.items():
                self.image_pairs.append(value)
                self.keys.append(key)
                
    def read_json(self,json_path,npz_path):
        dict_1 = {}
          
        with open(json_path, 'r') as load_f:
            json_data = json.load(load_f)
            dict_1['image_height'] = json_data['imageHeight']
            dict_1['image_width'] = json_data["imageWidth"]
            instances=json_data['instance']
            num_instance=len(instances)
            if num_instance>1:
                idx=np.random.randint(0,num_instance)
            else:
                idx=0

            box_list=[]
            shapes=instances[idx]
            for shape in shapes:
                if shape["shape_type"]=="rectangle":
                    box_list.append(shape["points"])    
            if os.path.exists(os.path.join(npz_path,str(idx+1)+".npz")):
                human_dict=load_smplx_data(os.path.join(npz_path,str(idx+1)+".npz"))
            else:
                human_dict={}
                human_dict['global_orient']=torch.zeros(1,3)
                human_dict['body_pose']=torch.zeros(21,3)
                human_dict['left_hand_pose']=torch.zeros(15,3)
                human_dict['right_hand_pose']=torch.zeros(15,3)
                human_dict['jaw_pose']=torch.zeros(1,3)
                human_dict['whole_body_pose']=torch.zeros(53,3)
                      
        load_f.close()

        h_o_x1=int(max(min(box_list[0][0][0],box_list[1][0][0])-20,0))
        h_o_x2=int(min(max(box_list[0][1][0],box_list[1][1][0])+20, dict_1['image_height'])) ####
        h_o_y1=int(max(min(box_list[0][0][1],box_list[1][0][1])-20,0))
        h_o_y2=int(min(max(box_list[0][1][1],box_list[1][1][1])+20, dict_1['image_width']))
        mask = np.zeros((dict_1['image_height'], dict_1['image_width']), dtype=np.float32)
        mask[h_o_y1:h_o_y2,h_o_x1:h_o_x2]=1
        mask = cv2.resize(mask, (self.crop_size, self.crop_size))
        mask=torch.from_numpy(mask)

        return human_dict,mask
        

    def __getitem__(self, index):

        dict_data={}
        interactive_img_path,non_interactive_img_path=self.image_pairs[index].split(",")
        interactive_img_path=os.path.join(self.image_root,"interactive",interactive_img_path)
        non_interactive_img_path=os.path.join(self.image_root,"non_interactive",non_interactive_img_path)
        interactive_img = Image.open(interactive_img_path).convert('RGB')
        key=self.keys[index]
        
        interactive_img= self.transform(interactive_img)

        non_interactive_img=Image.open(non_interactive_img_path).convert("RGB")
        non_interactive_img=self.transform(non_interactive_img)
        aff,obj,img_name=interactive_img_path.split("/")[-3],interactive_img_path.split("/")[-2],interactive_img_path.split("/")[-1]
        
        npz_path=os.path.join(self.mesh_root,aff,obj,img_name[:-4],"smplx")
        json_path=os.path.join(self.label_root,aff,obj,img_name[:-3]+"json")
        human_dict,h_o_mask=self.read_json(json_path=json_path,npz_path=npz_path)
       

        dict_data['interactive_image']=interactive_img
        dict_data['h_o_mask']=h_o_mask
        
        dict_data['human_dict']=human_dict
        dict_data['non_interactive_image']=non_interactive_img
        dict_data['key']=key
        
        return dict_data


    def __len__(self):
        return len(self.image_pairs)




def gen_heatmap(json_data, k_ratio=3,crop_size=224,flag="val"):
    k_size = int(np.sqrt(json_data['image_height'] * json_data['image_width']) / k_ratio)
    if k_size % 2 == 0:
        k_size += 1
   
    heatmap1 = np.zeros((json_data['image_height'], json_data['image_width']), dtype=np.float32)
    point1s = json_data['point1']
    for point in point1s:
        x = point[1]
        y = point[0]
        row = int(x)
        col = int(y)
        try:
            heatmap1[row, col] += 1.0
        except:
            # resize pushed it out of bounds somehow
            
            row = min(max(row, 0), json_data['image_height'] - 1)
            col = min(max(col, 0), json_data['image_width'] - 1)
            heatmap1[row, col] += 1.0
    heatmap1 = cv2.GaussianBlur(heatmap1, (k_size, k_size), 0)
    heatmap1 = (heatmap1 - np.min(heatmap1)) / (np.max(heatmap1) - np.min(heatmap1) + 1e-10)
    heatmap1 = cv2.resize(heatmap1, (crop_size, crop_size))

    #### heatmap2
    heatmap2 = np.zeros((json_data['image_height'], json_data['image_width']), dtype=np.float32)
    point2s = json_data['point2']
    for point in point2s:
        x = point[1]
        y = point[0]
        row = int(x)
        col = int(y)
        try:
            heatmap2[row, col] += 1.0
        except:
            # resize pushed it out of bounds somehow
            row = min(max(row, 0), json_data['image_height'] - 1)
            col = min(max(col, 0), json_data['image_width'] - 1)
            heatmap2[row, col] += 1.0
    heatmap2 = cv2.GaussianBlur(heatmap2, (k_size, k_size), 0)
    heatmap2 = (heatmap2 - np.min(heatmap2)) / (np.max(heatmap2) - np.min(heatmap2) + 1e-10)
    heatmap2 = cv2.resize(heatmap2, (crop_size, crop_size))

    #### heatmap3
    heatmap3 = np.zeros((json_data['image_height'], json_data['image_width']), dtype=np.float32)
    point3s = json_data['point3']
    for point in point3s:
        x = point[1]
        y = point[0]
        row = int(x)
        col = int(y)
        try:
            heatmap3[row, col] += 1.0
        except:
            # resize pushed it out of bounds somehow
            row = min(max(row, 0), json_data['image_height'] - 1)
            col = min(max(col, 0), json_data['image_width'] - 1)
            heatmap3[row, col] += 1.0
    heatmap3 = cv2.GaussianBlur(heatmap3, (k_size, k_size), 0)
    heatmap3 = (heatmap3 - np.min(heatmap3)) / (np.max(heatmap3) - np.min(heatmap3) + 1e-10)
    heatmap3 = cv2.resize(heatmap3, (crop_size, crop_size))

    #### heatmap4
    heatmap4 = np.zeros((json_data['image_height'], json_data['image_width']), dtype=np.float32)
    point4s = json_data['point4']
    for point in point4s:
        x = point[1]
        y = point[0]
        row = int(x)
        col = int(y)
        try:
            heatmap4[row, col] += 1.0
        except:
            # resize pushed it out of bounds somehow
            row = min(max(row, 0), json_data['image_height'] - 1)
            col = min(max(col, 0), json_data['image_width'] - 1)
            heatmap4[row, col] += 1.0
    heatmap4 = cv2.GaussianBlur(heatmap4, (k_size, k_size), 0)
    heatmap4 = (heatmap4 - np.min(heatmap4)) / (np.max(heatmap4) - np.min(heatmap4) + 1e-10)
    heatmap4 = cv2.resize(heatmap4, (crop_size, crop_size))

    heatmap5 = np.zeros((json_data['image_height'], json_data['image_width']), dtype=np.float32)
    point5s = json_data['point5']
    for point in point5s:
        x = point[1]
        y = point[0]
        row = int(x)
        col = int(y)
        try:
            heatmap5[row, col] += 1.0
        except:
            # resize pushed it out of bounds somehow
            row = min(max(row, 0), json_data['image_height'] - 1)
            col = min(max(col, 0), json_data['image_width'] - 1)
            heatmap5[row, col] += 1.0

    heatmap5 = cv2.GaussianBlur(heatmap5, (k_size, k_size), 0)
    heatmap5 = (heatmap5 - np.min(heatmap5)) / (np.max(heatmap5) - np.min(heatmap5) + 1e-10)
    heatmap5 = cv2.resize(heatmap5, (crop_size, crop_size))

    heatmap6 = np.zeros((json_data['image_height'], json_data['image_width']), dtype=np.float32)
    point6s = json_data['point6']
    for point in point6s:
        x = point[1]
        y = point[0]
        row = int(x)
        col = int(y)
        try:
            heatmap6[row, col] += 1.0
        except:
            # resize pushed it out of bounds somehow
            row = min(max(row, 0), json_data['image_height'] - 1)
            col = min(max(col, 0), json_data['image_width'] - 1)
            heatmap6[row, col] += 1.0
    heatmap6 = cv2.GaussianBlur(heatmap6, (k_size, k_size), 0)
    heatmap6 = (heatmap6 - np.min(heatmap6)) / (np.max(heatmap6) - np.min(heatmap6) + 1e-10)
    heatmap6 = cv2.resize(heatmap6, (crop_size, crop_size))

    heatmap7 = np.zeros((json_data['image_height'], json_data['image_width']), dtype=np.float32)
    point7s = json_data['point7']
    for point in point7s:
        x = point[1]
        y = point[0]
        row = int(x)
        col = int(y)
        try:
            heatmap7[row, col] += 1.0
        except:
            # resize pushed it out of bounds somehow
            row = min(max(row, 0), json_data['image_height'] - 1)
            col = min(max(col, 0), json_data['image_width'] - 1)
            heatmap7[row, col] += 1.0
    heatmap7 = cv2.GaussianBlur(heatmap7, (k_size, k_size), 0)
    heatmap7 = (heatmap7 - np.min(heatmap7)) / (np.max(heatmap7) - np.min(heatmap7) + 1e-10)
    heatmap7 = cv2.resize(heatmap7, (crop_size, crop_size))

    if flag=="train":
        map1=torch.from_numpy(heatmap1)
        map2=torch.from_numpy(heatmap2)
        map3=torch.from_numpy(heatmap3)
        map4=torch.from_numpy(heatmap4)
        map5=torch.from_numpy(heatmap5)
        map6=torch.from_numpy(heatmap6)
        map7=torch.from_numpy(heatmap7)

        return [map1,map2,map3,map4,map5,map6,map7]
    else:
        return [heatmap1, heatmap2, heatmap3, heatmap4, heatmap5, heatmap6,heatmap7]

def load_smplx_data(npz_path):
    npz_data = np.load(npz_path)
    human_dict={}
    human_dict['global_orient']=torch.from_numpy(npz_data["global_orient"])
    human_dict['body_pose']=torch.from_numpy(npz_data['body_pose'])
    human_dict['left_hand_pose']=torch.from_numpy(npz_data['left_hand_pose'])
    human_dict['right_hand_pose']=torch.from_numpy(npz_data['right_hand_pose'])
    human_dict['jaw_pose']=torch.from_numpy(npz_data['jaw_pose'])
    human_dict['whole_body_pose']=torch.cat((human_dict['global_orient'],human_dict['body_pose'],human_dict['left_hand_pose'],human_dict['right_hand_pose'],human_dict['jaw_pose']),dim=0)
    
    return human_dict
