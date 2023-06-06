
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
            fuzzy_m,
            logger,
            local_steps,
            tune_locally=False
    ):
        super(FuzzyClient, self).__init__(
            learners_ensemble=learners_ensemble,
            train_iterator=train_iterator,
            val_iterator=val_iterator,
            test_iterator=test_iterator,
            logger=logger,
            local_steps=local_steps,
            tune_locally=tune_locally
        )

        assert self.n_learners == 1, "FuzzyClient only supports single learner."
        self.only_with_global = False
        self.fuzzy_m=fuzzy_m

    def update_membership_loss(self,membership_mat, losses, client_id):

        membership_mat[client_id,:3]=F.softmax(losses.T,dim=1).T.mean(dim=1)
    # @calc_exec_time(calc_time=True)
    # @memory_profiler
    def update_membership_mat(self, membership_mat, cluster_params, client_params, client_id, global_fixed_m, fuzzy_m):
        n_clusters = cluster_params.size(0)
        
        self.fuzzy_m = np.ones((n_clusters,))/n_clusters 

        if global_fixed_m == True:
            p = float(2 / (fuzzy_m - 1))
        else :
            p = float(2 / (self.fuzzy_m - 1))
   
        distances = torch.zeros(n_clusters, device=client_params.device)
        with torch.no_grad():
            for i in range(n_clusters):
                diff = client_params - cluster_params[i]
                distances[i]= torch.norm(diff) 

            print("distances",distances)
            distances =distances-0.8*distances.min()
            print("distances after min",distances)

            # membership_mat[client_id]=F.softmax(distances,dim=0).T
            dens=[]
            for cluster_id in range(n_clusters):
                den = 0.0
                for j in range(n_clusters):
                    den += (distances[cluster_id] / distances[j])  ** p
                    if den>1.e+6: den=1.e+6
                dens.append(den)
                membership_mat[client_id, cluster_id] = 1.0 / den
            print("dens", dens)
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