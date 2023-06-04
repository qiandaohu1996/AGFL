from typing import List, OrderedDict, Tuple, Union
import warnings

import torch
import torch.nn as nn

from utils.my_profiler import calc_exec_time

calc_time=True

# @calc_exec_time(calc_time=calc_time)
def average_learners(
        learners,
        target_learner,
        weights=None,
        average_params=True,
        average_gradients=False):
    """
    Compute the average of a list of learners_ensemble and store it into learner

    :param learners:
    :type learners: List[Learner]
    :param target_learner:
    :type target_learner: Learner
    :param weights: tensor of the same size as learners_ensemble, having values between 0 and 1, and summing to 1,
                    if None, uniform learners_weights are used
    :param average_params: if set to true the parameters are averaged; default is True
    :param average_gradients: if set to true the gradient are also averaged; default is False
    :type weights: torch.Tensor

    """
    if not average_params and not average_gradients:
        return

    if weights is None:
        n_learners = len(learners)
        weights = (1 / n_learners) * torch.ones(n_learners, device=learners[0].device)

    else:
        weights = weights.to(learners[0].device)

    target_state_dict = target_learner.model.state_dict(keep_vars=True)

    for key in target_state_dict:

        if target_state_dict[key].data.dtype == torch.float32:

            if average_params:
                target_state_dict[key].data.zero_()

            if average_gradients:
                target_state_dict[key].grad = target_state_dict[key].data.clone()
                target_state_dict[key].grad.data.zero_()

            for learner_id, learner in enumerate(learners):
                state_dict = learner.model.state_dict(keep_vars=True)

                if average_params:
                    target_state_dict[key].data += weights[learner_id] * state_dict[key].data.clone()

                if average_gradients:
                    if state_dict[key].grad is not None:
                        target_state_dict[key].grad += weights[learner_id] * state_dict[key].grad.clone()
                    elif state_dict[key].requires_grad:
                        warnings.warn(
                            "trying to average_gradients before back propagation,"
                            " you should set `average_gradients=False`."
                        )

        else:
            # tracked batches
            target_state_dict[key].data.zero_()
            for learner_id, learner in enumerate(learners):
                state_dict = learner.model.state_dict()
                target_state_dict[key].data += state_dict[key].data.clone()


@calc_exec_time(calc_time=calc_time)
def fuzzy_average_cluster_model(
    client_models,
    cluster_models,
    membership_mat,  # use membership matrix instead of weights
    fuzzy_m,  # add fuzzy parameter m
    clients_weights,  # add clients_weights parameter
    average_params=True,
    average_gradients=False
): 
    clients_weights = clients_weights.to(membership_mat.device)
    # print("membership_mat",membership_mat[2:5])
 
    n_clusters=len(cluster_models)

    for cluster_id in range(n_clusters):
        target_state_dict = cluster_models[cluster_id].state_dict(keep_vars=True)

        for key in target_state_dict:

            if target_state_dict[key].data.dtype == torch.float32:

                if average_params:
                    target_state_dict[key].data.zero_()

                if average_gradients:
                    target_state_dict[key].grad = target_state_dict[key].data.clone()
                    target_state_dict[key].grad.data.zero_()

                for client_id, model in enumerate(client_models):
                    state_dict = model.state_dict(keep_vars=True)
                    print("membership_mat ",(membership_mat))

                    # print("membership_mat len",(membership_mat[client_id]))

                    print("clients_weights ", clients_weights)
                    membership_val = (membership_mat[client_id][cluster_id] * clients_weights[client_id]) ** fuzzy_m 
                    if average_params:
                        target_state_dict[key].data += ( membership_val * state_dict[key].data.clone())

                    if average_gradients:
                        if state_dict[key].grad is not None:
                            target_state_dict[key].grad += ( membership_val * state_dict[key].grad.clone())
                        elif state_dict[key].requires_grad:
                            warnings.warn(
                                "trying to average_gradients before back propagation,"
                                " you should set `average_gradients=False`."
                            )

            else:
                target_state_dict[key].data.zero_()
                for client_id, model in enumerate(client_models):
                    state_dict = model.state_dict()
                    target_state_dict[key].data += ( state_dict[key].data.clone())

    for cluster_id in range(n_clusters):
    
        target_state_dict = cluster_models[cluster_id].state_dict(keep_vars=True)
        # normalize each parameter in target_state_dict
        for key in target_state_dict:
            total_membership_val = torch.sum((membership_mat[:,cluster_id] * clients_weights) ** fuzzy_m)
            if target_state_dict[key].data.dtype == torch.float32:
                if average_params:
                    target_state_dict[key].data /= ( total_membership_val)
                if average_gradients:
                    if state_dict[key].grad is not None:
                        target_state_dict[key].grad.data /= ( total_membership_val)
                    elif state_dict[key].requires_grad:
                        warnings.warn(
                            "trying to average_gradients before back propagation,"
                            " you should set `average_gradients=False`."
                        )


@calc_exec_time(calc_time=calc_time)
def fuzzy_average_client_model(
        cluster_models,
        client_models,
        membership_mat,
        average_params=True,
        average_gradients=False):

    n_clients = len(client_models)
    for client_id in range(n_clients):
    
        target_state_dict = client_models[client_id].state_dict(keep_vars=True)

        for key in target_state_dict:

            if target_state_dict[key].data.dtype == torch.float32:

                if average_params:
                    target_state_dict[key].data.zero_()

                if average_gradients:
                    target_state_dict[key].grad = target_state_dict[key].data.clone()
                    target_state_dict[key].grad.data.zero_()

                for cluster_id, model in enumerate(cluster_models):
                    state_dict = model.state_dict(keep_vars=True)
                    membership_val = membership_mat[client_id][cluster_id]
                    if average_params:
                        target_state_dict[key].data += membership_val * state_dict[key].data.clone()

                    if average_gradients:
                        if state_dict[key].grad is not None:
                            target_state_dict[key].grad += membership_val * state_dict[key].grad.clone()
                        elif state_dict[key].requires_grad:
                            warnings.warn(
                                "trying to average_gradients before back propagation,"
                                " you should set `average_gradients=False`."
                            )

            else:
                target_state_dict[key].data.zero_()
                for cluster_id, model in enumerate(cluster_models):
                
                    state_dict = model.state_dict()
                    target_state_dict[key].data += (state_dict[key].data.clone())
 

def get_param_list(models):
        """
        get `models` parameters as a unique flattened tensor
        :return: torch.tensor
        """
        param_list = torch.stack(
            [torch.cat([param.flatten() for param in model.parameters()]) for model in models])

        return param_list

@calc_exec_time(calc_time=calc_time)
def partial_average(learners, average_learner, alpha):
    """
    performs a step towards aggregation for learners, i.e.

    .. math::
        \forall i,~x_{i}^{k+1} = (1-\alpha) x_{i}^{k} + \alpha \bar{x}^{k}

    :param learners:
    :type learners: List[Learner]
    :param average_learner:
    :type average_learner: Learner
    :param alpha:  expected to be in the range [0, 1]
    :type: float

    """
    source_state_dict = average_learner.model.state_dict()

    target_state_dicts = [learner.model.state_dict() for learner in learners]

    for key in source_state_dict:
        if source_state_dict[key].data.dtype == torch.float32:
            for target_state_dict in target_state_dicts:
                target_state_dict[key].data =\
                    (1-alpha) * target_state_dict[key].data + alpha * source_state_dict[key].data
                # print(f"key: {key}, target_state_dict[key].data: {target_state_dict[key].data}, source_state_dict[key].data: {source_state_dict[key].data}")
         

def apfl_partial_average(personal_learner, global_learner, alpha):

    source_state_dict = global_learner.model.state_dict()
    target_state_dict = personal_learner.model.state_dict() 

    for key in source_state_dict:
        if source_state_dict[key].data.dtype == torch.float32:
            target_state_dict[key].data = \
                alpha *   target_state_dict[key].data + (1-alpha) * source_state_dict[key].data
            
 
def differentiate_learner(target, reference_state_dict, coeff=1.):
    """
    set the gradient of the model to be the difference between `target` and `reference` multiplied by `coeff`

    :param target:
    :type target: Learner
    :param reference_state_dict:
    :type reference_state_dict: OrderedDict[str, Tensor]
    :param coeff: default is 1.
    :type: float

    """
    target_state_dict = target.model.state_dict(keep_vars=True)

    for key in target_state_dict:
        if target_state_dict[key].data.dtype == torch.float32:

            target_state_dict[key].grad = \
                coeff * (target_state_dict[key].data.clone() - reference_state_dict[key].data.clone())

def init_nn(nn, val=0.):
    for para in nn.parameters():
        para.data.fill_(val)
    return nn

def get_param_tensor(model):
    """
    get `model` parameters as a unique flattened tensor
    :return: torch.tensor

    """
    param_list = []
    for param in model.parameters():
        param_list.append(param.data.view(-1, ))

    return torch.cat(param_list)

def copy_model(target, source):
    """
    Copy learners_weights from target to source
    :param target:
    :type target: nn.Module
    :param source:
    :type source: nn.Module
    :return: None

    """
    target.load_state_dict(source.state_dict())


def simplex_projection(v, s=1):
    """
    Compute the Euclidean projection on a positive simplex
    Solves the optimisation problem (using the algorithm from [1]):

    .. math::
        min_w 0.5 * || w - v ||_2^2,~s.t. \sum_i w_i = s, w_i >= 0

    Parameters
    ----------
    v: (n,) torch tensor,
       n-dimensional vector to project
    s: int, optional, default: 1,
       radius of the simplex

    Returns
    -------
    w: (n,) torch tensor,
       Euclidean projection of v on the simplex

    Notes
    -----
    The complexity of this algorithm is in O(n log(n)) as it involves sorting v.

    References
    ----------
    [1] Wang, Weiran, and Miguel A. Carreira-Perpinán. "Projection
        onto the probability simplex: An efficient algorithm with a
        simple proof, and an application." arXiv preprint
        arXiv:1309.1541 (2013)
        https://arxiv.org/pdf/1309.1541.pdf

    """

    assert s > 0, "Radius s must be strictly positive (%d <= 0)" % s
    n, = v.shape

    u, _ = torch.sort(v, descending=True)

    cssv = torch.cumsum(u, dim=0)

    rho = int(torch.nonzero(u * torch.arange(1, n + 1) > (cssv - s))[-1][0])

    lambda_ = - float(cssv[rho] - s) / (1 + rho)

    w = v + lambda_

    w = (w * (w > 0)).clip(min=0)

    return w


    # def trainable_params(src: Union[OrderedDict[str, torch.Tensor], torch.nn.Module], requires_name=False
    # ) -> Union[List[torch.Tensor], Tuple[List[str], List[torch.Tensor]]]:
    #     parameters = []
    #     keys = []
    #     if isinstance(src, OrderedDict):
    #         for name, param in src.items():
    #             if param.requires_grad:
    #                 parameters.append(param)
    #                 keys.append(name)
    #     elif isinstance(src, torch.nn.Module):
    #         for name, param in src.state_dict(keep_vars=True).items():
    #             if param.requires_grad:
    #                 parameters.append(param)
    #                 keys.append(name)

    #     if requires_name:
    #         return keys, parameters
    #     else:
    #         return parameters