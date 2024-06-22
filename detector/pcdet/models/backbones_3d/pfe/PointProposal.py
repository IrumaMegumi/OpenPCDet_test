# Point Proposal Network, source points are painted
# 建议分阶段独立训练
import torch
import torch.nn as nn 
import torch.nn.functional as F
import numpy as np 
from torch.autograd import Variable
import time

class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k*k)
        self.relu = nn.ReLU()

        self.bn1 = nn.InstanceNorm1d(64)
        self.bn2 = nn.InstanceNorm1d(128)
        self.bn3 = nn.InstanceNorm1d(1024)
        self.bn4 = nn.InstanceNorm1d(512)
        self.bn5 = nn.InstanceNorm1d(256)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1,self.k*self.k).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x

class PointNetfeat(nn.Module):
    def __init__(self, global_feat = True, feature_transform = False):
        super(PointNetfeat, self).__init__()
        self.stn = STNkd(k=8)
        self.conv1 = torch.nn.Conv1d(8, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.InstanceNorm1d(64)
        self.bn2 = nn.InstanceNorm1d(128)
        self.bn3 = nn.InstanceNorm1d(1024)
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=64)

    def forward(self, x):
        n_pts = x.size()[2]
        trans = self.stn(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans)
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))
        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2,1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2,1)
        else:
            trans_feat = None

        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        if self.global_feat:
            return x
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, n_pts)
            return torch.cat([x, pointfeat], 1)
      
#关键点建议网络，注意Loss的处理
#可能需要做好多阶段训练的准备
class PointProposalNet(nn.Module):
    '''
        num_object_points: 你从原始点云中初筛的点数量
        num_keypoints: 采样关键点数量
    '''
    def __init__(self,num_object_points,num_keypoints,global_feat=False):
        super(PointProposalNet,self).__init__()
        self.feature_extraction_module=PointNetfeat(global_feat=global_feat)
        self.num_object_points=num_object_points
        self.num_keypoints=num_keypoints
        if self.num_object_points<self.num_keypoints:
            raise ValueError(f"num object points should be more than num keypoints,but your num object points is {self.num_object_points}, your num key points is {self.num_keypoints}")
        else:
            pass
        self.W_gate = nn.Linear(1088, 1088)
        self.b_gate = nn.Parameter(torch.zeros(1088))
        self.W_fc = nn.Linear(1088, 1088)
        #门控函数
        self.gate_function=nn.Sigmoid()
        
        self.evaluate_layer_1=nn.Linear(1088,512)
        self.instanceNorm_1=nn.InstanceNorm1d(512)
        self.evaluate_layer_2=nn.Linear(512,256)
        self.instanceNorm_2=nn.InstanceNorm1d(256)
        self.evaluate_layer_3=nn.Linear(256,1)
        self.dropout = nn.Dropout(p=0.3)
        self.activate_func_eval=nn.ReLU()
    def forward(self,batch_dict,is_training=False):
        #input shape:[batch(should be 1), num_points, num_point_features+1]
        #TODO: 对painted points的squeeze放在外面，我不太建议放在里面
        painted_points=batch_dict['painted_points']
        painted_points=painted_points[:,:,1:]
        painted_points=painted_points.squeeze(0)
        _, indices = torch.sort(painted_points[:, 4])
        sorted_points = painted_points[indices]
        object_points= sorted_points[:self.num_object_points, :]
        #object_points为初步筛选后的点，数量为6000
        object_points= object_points.unsqueeze(0)
        object_points= object_points.permute(0,2,1)
        object_point_features=self.feature_extraction_module(object_points)
        object_point_features=object_point_features.permute(0,2,1)
        #x to context fusion:[batch,num_points,num_point_features]
        #TODO: don't forget to concat it to the original point features
        # 调制特征
        g = self.gate_function(self.W_gate(object_point_features) + self.b_gate)
        # 上下文门控特征
        result_features = g * self.W_fc(object_point_features)
        scores_1=self.activate_func_eval(self.instanceNorm_1(self.evaluate_layer_1(result_features)))
        scores_2=self.activate_func_eval(self.instanceNorm_2(self.dropout(self.evaluate_layer_2(scores_1))))
        scores_final=self.evaluate_layer_3(scores_2)
        scores_final=scores_final.squeeze(0)
        _, indices = torch.sort(scores_final[:, 0])
        object_points=object_points.permute(0,2,1).squeeze(0)
        sorted_points = object_points[indices]
        keypoints= sorted_points[:self.num_keypoints, :]
        keypoints= keypoints.unsqueeze(0)
        if is_training==True:
            train_loss, tb_dict, disp_dict=self.calculate_loss(keypoints,batch_dict)
            return train_loss, tb_dict
        #返回采样点和采样时提取的原始点特征，后续不确定是否备用
        return keypoints,result_features
    #先写好训练的代码，然后再写loss
    def calculate_loss(self, keypoints,batch_dict):
        disp_dict={}
        sample_loss, tb_dict=self.calculate_sample_loss(keypoints, batch_dict)
        task_loss, tb_dict =self.calculate_task_loss(keypoints, batch_dict, tb_dict)
        loss= 0.3*sample_loss+0.7*task_loss
        return loss, tb_dict, disp_dict

    #点到各个框中心最小的smooth_l1 loss总和除以采样点总数
    def calculate_sample_loss(self, keypoints, batch_dict):
        pass
    
    #近远处点的比例和标准比例差的绝对值
    def calculate_task_loss(self, keypoints, batch_dict, tb_dict):
        pass


if __name__=='__main__':
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    pointfeat = PointProposalNet(num_object_points=6000,num_keypoints=2048,global_feat=False)
    pointfeat=pointfeat.to(device)
    while(1):
        sim_data = torch.rand(1,30000,9)
        sim_data=sim_data.to(device)
        start=time.time()
        keypoints,keypointfeat= pointfeat(sim_data, is_training=True)
        end=time.time()
        run_time=(end-start)*1000
        print('point feat', keypoints.size())
        print(f"{run_time}ms")