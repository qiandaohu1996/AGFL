
from copy import deepcopy
import numpy as np
import torch.nn.functional as F

from client import Client
from utils.torch_utils import *
from utils.constants import *

from utils.datastore import *

class FuzzyClient(Client): 
    

    def __init__(
            self,
            learners_ensemble,
            train_iterator,
            val_iterator,
            test_iterator,
            idx,
            trans,
            initial_fuzzy_m,
            fuzzy_m_scheduler,
            logger,
            local_steps,
            tune_locally=False
    ):
        super(FuzzyClient, self).__init__(
            learners_ensemble=learners_ensemble,
            train_iterator=train_iterator,
            val_iterator=val_iterator,
            test_iterator=test_iterator,
            idx=idx,
            logger=logger,
            local_steps=local_steps,
            tune_locally=tune_locally
        )

        assert self.n_learners == 1, "FuzzyClient only supports single learner."
        self.only_with_global = False
        self.fuzzy_m=initial_fuzzy_m
        # utils.py line 484
        self.fuzzy_m_scheduler=fuzzy_m_scheduler
        self.trans=trans
        self.previous_membership_vec = None

    def init_membership_vector(self,n_clusters):
        membership_vector = torch.rand(n_clusters, )
        membership_vector = membership_vector / membership_vector.sum(dim=0, keepdim=True)
        return membership_vector  

    # @calc_exec_time(calc_time=True)
    def update_membership(self, membership_mat, cluster_params, client_params, client_id, global_fixed_m, fuzzy_m,momentum=None):
        n_clusters = cluster_params.size(0)
        
        # self.fuzzy_m = np.ones((n_clusters,))/n_clusters 
        eps=1e-10
        if global_fixed_m == True:
            p = float(2 / (fuzzy_m - 1))
        else:
            p = float(2 / (self.fuzzy_m - 1))
        distances = torch.zeros(n_clusters, device=client_params.device)
        # dens= deepcopy(distances)
        with torch.no_grad():
            for i in range(n_clusters):
                diff = client_params - cluster_params[i]
                distances[i]= torch.norm(diff) 
             
            # self.torch_display("distances ", distances)
            distances = (distances - self.trans * distances.min()) +eps
            # distances = (distances - self.trans * distances.min())/ distances.std() 

            self.torch_display(f"distances after -{self.trans}min  ", distances)
 
            for cluster_id in range(n_clusters):
                den = 0.0
                for j in range(n_clusters):
                    if distances[j] > eps:  # 如果距离大于0
                        den += (distances[cluster_id] / distances[j]) ** p
                    else:  # 如果距离等于0
                        den += 1 if cluster_id == j else 0
                    den = torch.clamp(den, max=1e10)

                # dens[cluster_id]= den
                membership_mat[client_id, cluster_id] = 1.0 /den  # avoid division by zero
            
            # self.torch_display("dens  ", dens)
            # print("previous_membership_vec  " ,self.previous_membership_vec)  
            if self.previous_membership_vec is not None:
                membership_mat[client_id] =  self.previous_membership_vec * momentum+ membership_mat[client_id]*(1-momentum)
                # print("membership_mat  ", membership_mat[client_id])  
                
                # membership_mat[client_id] /= membership_mat[client_id].sum()

            self.previous_membership_vec = deepcopy(membership_mat[client_id])
            # print("membership_mat client ",client_id,"  ", membership_mat[client_id])    

            torch.cuda.empty_cache()

        return membership_mat


    def update_membership_loss(self,membership_mat, losses, client_id):

        membership_mat[client_id,:3]=F.softmax(losses.T,dim=1).T.mean(dim=1)

    # @calc_exec_time(calc_time=True)
    # @memory_profiler
    def update_membership_original(self, membership_mat, cluster_params, client_params, client_id, global_fixed_m, fuzzy_m):
        n_clusters = cluster_params.size(0)
        
        # self.fuzzy_m = np.ones((n_clusters,))/n_clusters 
        eps=1e-10
        if global_fixed_m == True:
            p = float(2 / (fuzzy_m - 1))
        else :
            p = float(2 / (self.fuzzy_m - 1))
   
        distances = torch.zeros(n_clusters, device=client_params.device)
        with torch.no_grad():
            for i in range(n_clusters):
                diff = client_params - cluster_params[i]
                distances[i]= torch.norm(diff) 
 
            print("distances  ", [f"{d:.3f}" for d in distances])

            distances = (distances - self.trans * distances.min()) +eps
            # distances = (distances - self.trans * distances.min())/(distances.max()-  distances.min())
            print(f"distances after -{self.trans}min    ", [f"{d:.3f}" for d in distances])

            dens=[]
            for cluster_id in range(n_clusters):
                den = 0.0
                for j in range(n_clusters):
                    if distances[j] > eps:  # 如果距离大于0
                        den += (distances[cluster_id] / distances[j]) ** p
                    else:  # 如果距离等于0
                        den += 1 if cluster_id == j else 0
                    den = torch.clamp(den, max=1e8)

                dens.append(den)
                membership_mat[client_id, cluster_id] = 1.0 /den  # avoid division by zero
                
            dens = torch.zeros(n_clusters, device='cuda:0')  # 初始化一个空张量

            for cluster_id in range(n_clusters):
                # 使用布尔掩码来区分距离大于eps的元素
                mask = distances > eps
                div = distances[cluster_id] / distances
                # 使用掩码将距离大于eps的元素替换为对应的计算值，否则为0
                div = torch.where(mask, div**p, torch.zeros_like(distances))
                # 对于距离等于0的元素，使用另一个布尔掩码来替换为1或0
                mask_zero = distances == 0
                div = torch.where(mask_zero, (cluster_id == torch.arange(n_clusters, device='cuda:0')).float(), div)
                # 累加所有的值得到den
                den = div.sum()
                den = torch.clamp(den, max=1e8)

                dens[cluster_id] = den
                membership_mat[client_id, cluster_id] = 1.0 / den
            

            # 打印转换后的列表
            # formatted_dens = [f"{d:.3f}" for d in dens]
            dens =dens.cpu().tolist()
            rounded_dens = [round(d, 3) for d in dens]
            print("dens ", rounded_dens)

            print("membership_mat client ",client_id,"  ", membership_mat[client_id])    
            torch.cuda.empty_cache()

        return membership_mat
    

    

    
    def update_membership_loss_momentum(self, membership_mat, cluster_params, client_params, client_id, global_fixed_m, fuzzy_m,momentum=None):
        n_clusters = cluster_params.size(0)
        
        # self.fuzzy_m = np.ones((n_clusters,))/n_clusters 
        eps=1e-10
        if global_fixed_m == True:
            p = float(2 / (fuzzy_m - 1))
        else:
            p = float(2 / (self.fuzzy_m - 1))
            
        momentum=0.8
        distances = torch.zeros(n_clusters, device=client_params.device)
        dens= deepcopy(distances)
        with torch.no_grad():
            for i in range(n_clusters):
                diff = client_params - cluster_params[i]
                distances[i]= torch.norm(diff) 
             
            self.torch_display("distances ", distances)
            distances = (distances - self.trans * distances.min()) +eps
            # distances = (distances - self.trans * distances.min())/(distances.max()-  distances.min())

            self.torch_display(f"distances after -{self.trans}min  ", distances)
 
            for cluster_id in range(n_clusters):
                den = 0.0
                for j in range(n_clusters):
                    if distances[j] > eps:  # 如果距离大于0
                        den += (distances[cluster_id] / distances[j]) ** p
                    else:  # 如果距离等于0
                        den += 1 if cluster_id == j else 0
                    den = torch.clamp(den, max=1e8)

                dens[cluster_id]= den
                membership_mat[client_id, cluster_id] = 1.0 /den  # avoid division by zero
            
            self.torch_display("dens  ", dens)

            if self.previous_membership_vec is not None:
                membership_mat[client_id] =  self.previous_membership_vec * momentum+ membership_mat[client_id]*(1-momentum)
                membership_mat[client_id] /= membership_mat[client_id].sum()

            self.previous_membership_vec = deepcopy(membership_mat[client_id])
            print("membership_mat client ",client_id,"  ", membership_mat[client_id])    

            torch.cuda.empty_cache()

        return membership_mat
    
    def torch_display(self,info, tensor):
            tensor =tensor.cpu().tolist()
            rounded_tensor = [round(d, 3) for d in tensor]
            print(info, rounded_tensor)

    def update_membership_cosine(self, membership_mat, cluster_params, client_params, client_id, global_fixed_m, fuzzy_m):
        n_clusters = cluster_params.size(0)
        self.fuzzy_m = np.ones((n_clusters,))/n_clusters 

        if global_fixed_m == True:
            p = float(2 / (fuzzy_m - 1))
        else :
            p = float(2 / (self.fuzzy_m - 1))
        # 预先计算所有距离
        distances = torch.zeros(n_clusters, device=self.learners_ensemble.device)
        with torch.no_grad():
            for i in range(n_clusters):
                # 添加额外的维度以计算余弦相似度
                cos_sim = F.cosine_similarity(client_params.unsqueeze(0), cluster_params[i].unsqueeze(0))
                distances[i] = (1.0 - cos_sim)
                # log_distances[i] = torch.log(distances[i]+1e-10)
            print("distances ",distances)

            distances = torch.clamp(distances, min=0)

            distances = (distances - self.trans*distances.min()) / (distances.max() - distances.min() + 1e-10) + 1e-10
            print("distances max-min ",distances)
            dens=[]
            for cluster_id in range(n_clusters):
                den = 0.0
                for j in range(n_clusters):
                    den += (distances[cluster_id] / distances[j]) ** p 
                    if den>1.e+6: den=1.e+6
                dens.append(den)
                membership_mat[client_id, cluster_id] = 1.0 / den
            
            formatted_dens = [f"{d:.3f}" for d in dens]

            print("dens", formatted_dens)
            print("membership_mat client ",client_id,"  ", membership_mat[client_id])   
            torch.cuda.empty_cache()

        return membership_mat


    def update_global_membership_mat(self, membership_mat, cluster_params, client_params, client_id, global_fixed_m, fuzzy_m):
        n_clusters = cluster_params.size(0)
        comm_global_flag=False
        self.fuzzy_m = np.ones((n_clusters,))/n_clusters 
        # global_fixed_m=True
        if global_fixed_m == False:
            p = float(2 / (self.fuzzy_m - 1))
        else :
            p = float(2 / (fuzzy_m - 1))

        distances = torch.zeros(n_clusters, device=client_params.device)
        with torch.no_grad():
            for i in range(n_clusters):
                diff = client_params - cluster_params[i]
                distances[i]= torch.norm(diff) 

            print("distances",distances)
            distances =distances-0.5*distances.min()
            print("distances after min",distances)

            # membership_mat[client_id]=F.softmax(distances,dim=0).T

            for cluster_id in range(n_clusters):
                den = 0.0
                for j in range(n_clusters):
                    den += (distances[cluster_id] / distances[j])  ** p
                    if den>1.e+6: den=1.e+6
                print("den ", den)
                membership_mat[client_id, cluster_id] = 1.0 / den
            if membership_mat[client_id,-1] > 0.95: 
                comm_global_flag=True

            for cluster_id in range(n_clusters-1):
                den = 0.0
                for j in range(n_clusters-1):
                    den += (distances[cluster_id] / distances[j])  ** p
                    if den>1.e+6: den=1.e+6
                print("den ", den)
                membership_mat[client_id, cluster_id] = 1.0 / den
            print("membership_mat client ",client_id,"  ", membership_mat[client_id])

        return comm_global_flag