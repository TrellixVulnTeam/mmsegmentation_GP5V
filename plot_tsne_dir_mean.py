
from sklearn import manifold
import matplotlib.pyplot as plt
import pickle
import os
# pickle.load()
import numpy as np
import pdb
import torch
import mmcv
# from tsnecuda import TSNE
import cv2

# model_dict = torch.load('work_dirs/fpn_twins_cascade_dpet_multi_memory_isaid/iter_160000.pth')
# query_feature = model_dict['state_dict']['decode_head.large_batch_queue.large_batch_queue'][::4]
# pkl_dir ='feature_vis_dpet_vaihingen_normalized'
# vis_dir = 'vis_tsne_cpu_vaihingen_train_dpet_normalized'
pkl_dir ='feature_vis_deeplab_vaihingen'
vis_dir = 'vis_tsne_cpu_vaihingen_test_deeplabv3_normalized'


if not os.path.exists(vis_dir):
    os.mkdir(vis_dir)
ann_dir = 'data/vaihingen/converted/ann_dir/val_split/'
colors = ['purple','blue','lime','green','yellow','red','deeppink','orange','cyan','limegreen','black']

# colors = ['purple','blue','red','green','yellow','lime','deeppink','orange','cyan','limegreen','black']

cls_features=[0]*6

for i in range(6):
    cls_features[i]=[]


for pkl in os.listdir(pkl_dir):
    pkl_name = os.path.join(pkl_dir,pkl)

    ann_name = os.path.join(ann_dir,pkl.replace('pkl','png'))
    ann_img_gt = mmcv.imread(ann_name)[:,:,0]
    ann_img_gt = cv2.resize(ann_img_gt,(128,128))

    pkl_data = mmcv.load(pkl_name)
    query_feature= pkl_data[1][0]
    if len(query_feature.shape)==3:
        num_channel = 512 
        query_feature = torch.reshape(query_feature.permute(1,2,0),(-1,num_channel))

    ann_img= pkl_data[0][0]
    ann_img = ann_img[2::4,2::4]
    ann_img = ann_img.reshape((-1))

    for i in range(6):
        if len(np.where(ann_img==i)[0])!=0:
            cls_feature = torch.mean(query_feature[np.where(ann_img==i)],dim=0)
            cls_features[i].append(cls_feature)

cls_label=[]
for i in range(6):
    print(i,len(cls_features[i]))
    cls_features[i]=torch.stack(cls_features[i])
    cls_label.append(torch.ones(len(cls_features[i]))*i)

cls_label = torch.cat(cls_label).numpy()
cls_features = torch.cat(cls_features).cpu().numpy()

tsne = manifold.TSNE(n_components=2,init='pca', random_state=1)
query_feature_tsne = tsne.fit_transform(cls_features)
plt.clf()

for i in range(6):
    cls_idx = np.where(cls_label==i)
    plt.scatter(query_feature_tsne[cls_idx, 0],  query_feature_tsne[cls_idx, 1],s=20,color = colors[i], marker='o')
# plt.savefig(vis_dir+'/'+pkl.split('.')[0]+'.png')
plt.savefig(vis_dir+'_all_'+'.png')