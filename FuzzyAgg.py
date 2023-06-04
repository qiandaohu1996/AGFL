
from copy import deepcopy

import numpy as np
# import numpy.linalg as LA

from sklearn.metrics import pairwise_distances
from sklearn.cluster import AgglomerativeClustering
from aggregator import Aggregator
from utils.my_profiler import calc_exec_time
from utils.torch_utils import *
# from utils.fuzzy_cluster import *
# from finch import FINCH
from torch.cuda.amp import autocast as autocast

from learners.learner import *
from learners.learners_ensemble import *

calc_time=True


class FuzzyGroupAggregator(Aggregator):

    def __init__(
            self,
            clients,
            global_learners_ensemble,
            log_freq,
            global_train_logger,
            global_test_logger,
            local_test_logger,
            communication_probability,
            single_batch_flag,
            fuzzy_m=2,
            pre_rounds=0,
            sampling_rate=1.,
            sample_with_replacement=False,
            test_clients=None,
            verbose=0,
            seed=None
    ):
        super(FuzzyGroupAggregator, self).__init__(
            clients=clients,
            global_learners_ensemble=global_learners_ensemble,
            log_freq=log_freq,
            global_train_logger=global_train_logger,
            global_test_logger=global_test_logger,
            local_test_logger=local_test_logger,
            communication_probability=communication_probability,
            single_batch_flag=single_batch_flag,
            sampling_rate=sampling_rate,
            sample_with_replacement=sample_with_replacement,
            test_clients=test_clients,
            verbose=verbose,
            seed=seed
        )

        self.single_batch_flag=single_batch_flag    
        self.fuzzy_m=fuzzy_m
        self.global_learner=self.global_learners_ensemble[0]
        self.membership_mat= None
        self.n_clusters=1

        self.pre_rounds=pre_rounds


    def init_membership_mat(self,n_clients,n_clusters):
        membership_mat = torch.rand(n_clients, n_clusters)
        membership_mat = membership_mat / membership_mat.sum(dim=1, keepdim=True)
        
        print("init membership_mat:", membership_mat[2:5])
        return membership_mat  
    
    def pre_train(self):

        for client_id, client in enumerate(self.clients):
            # print("client_id ", client_id)

            client.step(self.single_batch_flag)

        clients_learners = [client.learners_ensemble[0] for client in self.clients]
        average_learners(clients_learners, self.global_learners_ensemble[0], weights=self.clients_weights)

        self.update_clients(self.clients)


    @calc_exec_time(calc_time=calc_time)
    def pre_clusting(self, n_clusters):
        print(f"\n============start clustring==============")
        clients_updates = np.zeros((self.n_clients, self.model_dim))
        for client_id, client in enumerate(self.clients):
            clients_updates[client_id] = client.step(self.single_batch_flag)  
        
        self.n_clusters = n_clusters
        print("clients_updates size",clients_updates.shape)
        similarities = pairwise_distances(clients_updates, metric="cosine")
        print("similarities size",similarities.shape)

        clustering = AgglomerativeClustering(
            n_clusters=self.n_clusters, metric="precomputed", linkage="complete")
        clustering.fit(similarities)
        
        self.clusters_indices = [np.argwhere(clustering.labels_ == i).flatten() for i in range(self.n_clusters)]
        # print("clusters_indices ",self.clusters_indices)

        print(f"=============cluster completed===============")

        print("\ndivided into {} clusters:".format(self.n_clusters))
        for i, cluster in enumerate(self.clusters_indices):
            print(f"cluster {i}: {cluster}")

        cluster_weights= torch.zeros(self.n_clusters)
        
        learners = [deepcopy(self.clients[0].learners_ensemble[0])
                     for _ in range(self.n_clusters)]
        for learner in learners:
            # init_nn(learner.model)
            learner.device=self.device
            
        self.cluster_learners= LearnersEnsemble(learners=learners, learners_weights=cluster_weights)

        for cluster_id, indices in enumerate(self.clusters_indices):
            cluster_weights[cluster_id]= self.clients_weights[indices].sum()

            cluster_clients = [self.clients[i] for i in indices]

            average_learners(
                learners=[client.learners_ensemble[0] for client in cluster_clients],
                target_learner=self.cluster_learners[cluster_id],
                weights=self.clients_weights[indices] /
                cluster_weights[cluster_id]
            )
            print("cluster_params ",cluster_id,"  ", get_param_tensor(self.cluster_learners[cluster_id].model)[10:15])
            
        self.membership_mat=self.init_membership_mat(self.n_clients,self.n_clusters+1)
        self.membership_mat = self.membership_mat.to(self.device)

        # print("init membership_mat ",self.membership_mat[2:5])
        
    def train(self):
        self.cluster_comm_clients_indices=list(range(self.n_clients))
        self.global_comm_clients_indices=[]

        self.cluster_comm_clients_weights=self.clients_weights
        self.cluster_comm_clients_membership_mat=torch.empty((),device=self.device)
        cluster_models = [learner.model for learner in self.cluster_learners]

        client_learners= [client.learners_ensemble[0] for client in self.clients ]
        cluster_comm_clients_learners = [self.clients[i].learners_ensemble[0] for i in self.cluster_comm_clients_indices]

        client_models = [learner.model for learner in client_learners]
        cluster_comm_client_models = [learner.model for learner in cluster_comm_clients_learners]

        with segment_timing("updating all clients' model"):
            for client_id, client in enumerate(self.clients):
                client.step(self.single_batch_flag)
            print(self.membership_mat[2:5])

        with segment_timing("updating all clients' membership matrices "):
        
            if self.c_round % 5 == 0 :
                cluster_models.append(self.global_learners_ensemble[0].model)
                cluster_params = get_param_list(cluster_models)

                for client_id, client in enumerate(self.clients):
                    client_params = get_param_tensor(client.learners_ensemble[0].model)
                    comm_global_flag=client.update_global_membership_mat(
                        client_params=client_params,
                        cluster_params=cluster_params,
                        membership_mat= self.membership_mat,
                        global_fixed_m=True,
                        fuzzy_m=self.fuzzy_m,
                        client_id=client_id
                        )
                    
                if comm_global_flag:
                    self.cluster_comm_clients_indices.remove(client_id)
                    self.global_comm_clients_indices.append(client_id)

                print("cluster_comm_clients length",len(self.cluster_comm_clients_indices))
                print("global_comm_clients length",len(self.global_comm_clients_indices))
                self.update_cluster_comm_client_weights()
             
                self.cluster_comm_clients_membership_mat= self.membership_mat[self.cluster_comm_clients_indices, :]
                
            else:
                cluster_params = get_param_list(cluster_models)
                # self.cluster_comm_clients=self.clients[:]
                # print(cluster_params.size())
                self.cluster_comm_clients=[self.clients[i] for i in self.cluster_comm_clients_indices]
                for client_id, client in enumerate(self.cluster_comm_clients):
                    client_params = get_param_tensor(client.learners_ensemble[0].model)
                    client.update_membership_mat(
                        client_params=client_params,
                        cluster_params=cluster_params,
                        membership_mat= self.cluster_comm_clients_membership_mat,
                        global_fixed_m=True,
                        fuzzy_m=self.fuzzy_m,
                        client_id=client_id
                        )

        with segment_timing("aggregate to get the cluster model "):
            fuzzy_average_cluster_model(
                client_models=cluster_comm_client_models,
                cluster_models=cluster_models,
                membership_mat= self.cluster_comm_clients_membership_mat,
                fuzzy_m=self.fuzzy_m,
                clients_weights=self.cluster_comm_clients_weights
                )
                
            print("after:", self.membership_mat[2:5])
            
        with segment_timing("aggregate to get the client model "):

            fuzzy_average_client_model(
                membership_mat= self.cluster_comm_clients_membership_mat,
                cluster_models=cluster_models,
                client_models=cluster_comm_client_models
                )

        with segment_timing("aggregate to get the global model "):
            average_learners(
                learners=client_learners,
                target_learner=self.global_learners_ensemble[0],
                weights=self.clients_weights
                )
            self.global_comm_clients=[self.clients[i] for i in self.global_comm_clients_indices]
            self.update_clients(self.global_comm_clients)
            
    def mix(self):

        
        if self.c_round<self.pre_rounds:
            self.pre_train()

        elif self.c_round==self.pre_rounds and self.n_clusters==1:
            self.pre_clusting(n_clusters=3)
            self.c_round -= 1
            
        else: 
            self.train()

        if self.c_round % self.log_freq == 0   or self.c_round==199:
            self.write_logs()
        if self.c_round % 30==0 or self.c_round==199:
            print(self.membership_mat)
        self.c_round += 1   

        if self.c_round==200:
            print("c_round==200")
        if self.c_round==201:
            print("c_round==201")

    def update_clients(self, clients):
        for client in clients:
            for learner in client.learners_ensemble:
                copy_model(target=learner.model,
                           source=self.global_learners_ensemble[0].model)

    def update_cluster_comm_client_weights(self):

        self.cluster_comm_clients_weights =\
            torch.tensor(
                [self.clients[i].n_train_samples for i in self.cluster_comm_clients_indices],
                dtype=torch.float32
            )
        self.cluster_comm_clients_weights = self.cluster_comm_clients_weights / self.cluster_comm_clients_weights.sum()

