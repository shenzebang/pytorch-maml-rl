import torch
from torch.nn.utils.convert_parameters import (vector_to_parameters,
                                               parameters_to_vector)
from torch.distributions.kl import kl_divergence

from maml_rl.utils.torch_utils import (weighted_mean, detach_distribution,
                                       weighted_normalize)
from maml_rl.utils.optimization import conjugate_gradient

from collections import OrderedDict


def W2(pi_1, pi_2):
    W2 = (pi_1.mean - pi_2.mean).pow(2).sum(2, keepdim=True) + (pi_1.stddev - pi_2.stddev).pow(2).sum(2, keepdim=True)
    return W2

def sigma_square(pi):
    return (pi.stddev).pow(2).sum(2, keepdim=True)

def named_parameters_to_vector(named_parameters):
    vec = []
    for (name, param) in named_parameters.items():

        vec.append(param.view(-1))
    return torch.cat(vec)


def vector_to_named_parameter_like(vector, named_parameter_example):
    pointer = 0
    params = OrderedDict()
    for name, param_ex in named_parameter_example:
        # The length of the parameter
        num_param = param_ex.numel()
        # Slice the vector, reshape it, and replace the old data of the parameter
        params[name] = vector[pointer:pointer + num_param].view_as(param_ex)

        # Increment the pointer
        pointer += num_param

    return params

class MetaLearnerNGLVCVPG(object):
    """Meta-learner

    The meta-learner is responsible for sampling the trajectories/episodes 
    (before and after the one-step adaptation), compute the inner loss, compute 
    the updated parameters based on the inner-loss, and perform the meta-update.

    [1] Chelsea Finn, Pieter Abbeel, Sergey Levine, "Model-Agnostic 
        Meta-Learning for Fast Adaptation of Deep Networks", 2017 
        (https://arxiv.org/abs/1703.03400)
    [2] Richard Sutton, Andrew Barto, "Reinforcement learning: An introduction",
        2018 (http://incompleteideas.net/book/the-book-2nd.html)
    [3] John Schulman, Philipp Moritz, Sergey Levine, Michael Jordan, 
        Pieter Abbeel, "High-Dimensional Continuous Control Using Generalized 
        Advantage Estimation", 2016 (https://arxiv.org/abs/1506.02438)
    [4] John Schulman, Sergey Levine, Philipp Moritz, Michael I. Jordan, 
        Pieter Abbeel, "Trust Region Policy Optimization", 2015
        (https://arxiv.org/abs/1502.05477)
    """
    def __init__(self, sampler, policy, baseline, gamma=0.95,
                 fast_lr=0.5, tau=1.0, device='cpu', verbose=False):
        self.sampler = sampler
        self.policy = policy
        self.baseline = baseline
        self.gamma = gamma
        self.fast_lr = fast_lr
        self.tau = tau
        self.to(device)
        # for adam optimizer
        self.v = 0
        self.sqr = 0
        self.t = 0
        # print debug information
        self.verbose = verbose

    def inner_loss_lvc(self, episodes, params=None):
        """Compute the inner loss for the one-step gradient update. The inner 
        loss is REINFORCE with baseline [2], computed on advantages estimated 
        with Generalized Advantage Estimation (GAE, [3]).
        """
        values = self.baseline(episodes)
        advantages = episodes.gae(values, tau=self.tau)
        advantages = weighted_normalize(advantages, weights=episodes.mask)

        pi = self.policy(episodes.observations, params=params)
        log_probs = pi.log_prob(episodes.actions)
        if log_probs.dim() > 2:
            log_probs = torch.sum(log_probs, dim=2)
        DiCE_LVC = torch.exp(log_probs - log_probs.detach())
        loss = -weighted_mean(DiCE_LVC * advantages, dim=0,
                              weights=episodes.mask)

        return loss

    def adapt_ng(self, episodes, first_order=False, max_kl=1e-3, cg_iters=20, cg_damping=1e-2,
             ls_max_steps=10, ls_backtrack_ratio=0.5):
        """Adapt the parameters of the policy network to a new task, from
        sampled trajectories `episodes`, with a one-step natural gradient update.
        """
        # Fit the baseline to the training episodes
        self.baseline.fit(episodes)
        # Get the loss on the training episodes
        loss_lvc = self.inner_loss_lvc(episodes)
        # Get the new parameters after a one-step natural gradient update
        grads = torch.autograd.grad(loss_lvc, self.policy.parameters())
        grads = parameters_to_vector(grads)

        # Compute the step direction with Conjugate Gradient
        hessian_vector_product = self.hessian_vector_product_ng(episodes,
                                                             damping=cg_damping)
        stepdir = conjugate_gradient(hessian_vector_product, grads,
                                     cg_iters=cg_iters).detach()
        if self.verbose:
            print(torch.norm(hessian_vector_product(stepdir) - grads) / torch.norm(grads))

        shs = 0.5 * (stepdir.dot(hessian_vector_product(stepdir)))
        lm = torch.sqrt(max_kl / shs)

        if self.verbose:
            print("learning rate {}".format(lm))
        stepdir_named = vector_to_named_parameter_like(stepdir, self.policy.named_parameters())
        step_size = lm.detach()
        params = OrderedDict()
        for (name, param) in self.policy.named_parameters():
            params[name] = param - step_size * stepdir_named[name]

        if self.verbose:
            # compute the kl divergence
            with torch.autograd.no_grad():
                pi = self.policy(episodes.observations, params=params)
                pi_old = self.policy(episodes.observations)
                kl = kl_divergence(pi_old, pi).mean()
                print(kl)


        return params, step_size, stepdir

    def sample(self, tasks, first_order=False, cg_iters = 20, cg_damping=1e-2):
        """Sample trajectories (before and after the update of the parameters) 
        for all the tasks `tasks`.
        """
        # print("sample")
        episodes = []
        kls = []
        W2D2s = []
        sss = []
        param_diffs = []
        curr_params = self.policy.parameters()
        curr_params_flat = parameters_to_vector(curr_params)
        for task in tasks:
            self.sampler.reset_task(task)
            train_episodes = self.sampler.sample(self.policy,
                gamma=self.gamma, device=self.device)

            params, _, _ = self.adapt_ng(train_episodes, first_order=first_order, cg_iters=cg_iters, cg_damping=cg_damping)

            # compute the kl divergence
            with torch.autograd.no_grad():
                pi = self.policy(train_episodes.observations, params=params)
                pi_old = self.policy(train_episodes.observations)
                kl = kl_divergence(pi_old, pi).mean()
                params_flat = named_parameters_to_vector(params)
                param_diff = torch.norm(params_flat - curr_params_flat)
                W2D2 = W2(pi, pi_old)
                ss = sigma_square(pi)

            kls.append(kl)
            W2D2s.append(W2D2)
            sss.append(ss)
            param_diffs.append(param_diff)

            valid_episodes = self.sampler.sample(self.policy, params=params,
                gamma=self.gamma, device=self.device)
            episodes.append((train_episodes, valid_episodes))
        return episodes, kls, param_diffs, W2D2s, sss

    def W2_ng(self, episodes):
        # episode is the train episode
        pi = self.policy(episodes.observations)
        pi_detach = detach_distribution(pi)
        mask = episodes.mask
        if episodes.actions.dim() > 2:
            mask = mask.unsqueeze(2)
        kl = weighted_mean(W2(pi_detach, pi), dim=0, weights=mask)
        return kl

    def hessian_vector_product_ng(self, episodes, damping=1e-2):
        """Hessian-vector product, based on the Perlmutter method."""
        def _product(vector):
            kl = self.W2_ng(episodes)
            grads = torch.autograd.grad(kl, self.policy.parameters(), create_graph=True)
            flat_grad_kl = parameters_to_vector(grads)

            grad_kl_v = torch.dot(flat_grad_kl, vector)
            grad2s = torch.autograd.grad(grad_kl_v, self.policy.parameters(), create_graph=True)
            flat_grad2_kl = parameters_to_vector(grad2s)

            return flat_grad2_kl + damping * vector
        return _product

    def step_adam(self, episodes, max_kl=1e-3, cg_iters=20, cg_damping=1e-2,
             ls_max_steps=10, ls_backtrack_ratio=0.5):
        """Meta-optimization step (ie. update of the initial parameters), based 
        on Trust Region Policy Optimization (TRPO, [4]).
        """
        # print("step")
        grads = self.compute_ng_gradient(episodes, cg_iters=cg_iters)

        old_params = parameters_to_vector(self.policy.parameters())
        update_params = self.adam_step(old_params, grads)
        vector_to_parameters(update_params, self.policy.parameters())

    def step_sgd(self, episodes, max_kl=1e-3, cg_iters=20, cg_damping=1e-2,
             ls_max_steps=10, ls_backtrack_ratio=0.5):
        """Meta-optimization step (ie. update of the initial parameters), based
        on Trust Region Policy Optimization (TRPO, [4]).
        """
        print("step")
        grads = self.compute_ng_gradient(episodes, cg_iters=cg_iters, cg_damping=cg_damping)

        old_params = parameters_to_vector(self.policy.parameters())
        update_params = old_params - 3e-2*grads
        vector_to_parameters(update_params, self.policy.parameters())

    def to(self, device, **kwargs):
        self.policy.to(device, **kwargs)
        self.baseline.to(device, **kwargs)
        self.device = device

    def compute_ng_gradient(self, episodes, max_kl=1e-3, cg_iters=20, cg_damping=1e-2,
             ls_max_steps=10, ls_backtrack_ratio=0.5):
        ng_grads = []
        for train_episodes, valid_episodes in episodes:
            params_adapt, step_size, stepdir = self.adapt_ng(train_episodes, cg_iters=cg_iters, cg_damping=cg_damping)

            # compute $grad = \nabla_x J^{lvc}(x) at x = \theta - \eta\UM(\theta)
            self.baseline.fit(valid_episodes)
            loss = self.inner_loss_lvc(valid_episodes, params=params_adapt)
            ng_grad_0 = torch.autograd.grad(loss, self.policy.parameters())  # no create graph
            ng_grad_0 = parameters_to_vector(ng_grad_0)

            # compute the inverse of Fisher matrix at x=\theta times $grad with Conjugate Gradient
            hessian_vector_product = self.hessian_vector_product_ng(train_episodes,
                                                                 damping=cg_damping)
            F_inv_grad = conjugate_gradient(hessian_vector_product, ng_grad_0,
                                         cg_iters=cg_iters*2)

            if self.verbose:
                print(torch.norm(hessian_vector_product(F_inv_grad) - ng_grad_0) / torch.norm(ng_grad_0))

            # compute $ng_grad_1 = \nabla^2 J^{lvc}(x) at x = \theta times $F_inv_grad
            # create graph for higher differential
            self.baseline.fit(train_episodes)
            loss = self.inner_loss_lvc(train_episodes)

            grad = torch.autograd.grad(loss, self.policy.parameters(), create_graph=True)
            grad = parameters_to_vector(grad)
            grad_F_inv_grad = torch.dot(grad, F_inv_grad.detach())
            ng_grad_1 = torch.autograd.grad(grad_F_inv_grad, self.policy.parameters())
            ng_grad_1 = parameters_to_vector(ng_grad_1)

            # compute $ng_grad_2 = the Jacobian of {F(x) U(\theta)} at x = \theta times $F_inv_grad
            hessian_vector_product = self.hessian_vector_product_ng(train_episodes, damping=cg_damping)
            F_U = hessian_vector_product(stepdir)
            ng_grad_2 = torch.autograd.grad(torch.dot(F_U, F_inv_grad.detach()), self.policy.parameters())
            ng_grad_2 = parameters_to_vector(ng_grad_2)
            ng_grad = ng_grad_0 - step_size * (ng_grad_1 - ng_grad_2)

            ng_grad = parameters_to_vector(ng_grad)
            ng_grads.append(ng_grad.view(len(ng_grad), 1))

        return torch.mean(torch.stack(ng_grads, dim=1), dim=[1, 2])

    def adam_step(self, params, grad, lr=5e-4):
        beta1 = 0.9
        beta2 = 0.999
        eps_stable = 1e-5

        self.v = beta1 * self.v + (1. - beta1) * grad
        self.sqr = beta2 * self.sqr + (1. - beta2) * grad**2
        self.t += 1

        v_bias_corr = self.v / (1. - beta1 ** self.t)
        sqr_bias_corr = self.sqr / (1. - beta2 ** self.t)

        div = lr * v_bias_corr / (torch.sqrt(sqr_bias_corr) + eps_stable)
        params_update = params - div
        return params_update

    ####################################################################################################################
    def compute_ng_gradient_test(self, episodes, max_kl=1e-3, cg_iters=20, cg_damping=1e-2,
             ls_max_steps=10, ls_backtrack_ratio=0.5):
        ng_grads = []
        for train_episodes, valid_episodes in episodes:
            params_adapt, step_size, _ = self.adapt_ng_test(train_episodes)

            # self.baseline.fit(valid_episodes)
            loss = self.inner_loss_lvc(valid_episodes, params=params_adapt)
            ng_grad_0 = torch.autograd.grad(loss, self.policy.parameters())  # no create graph
            ng_grad_0 = parameters_to_vector(ng_grad_0)

            self.baseline.fit(train_episodes)
            loss = self.inner_loss_lvc(train_episodes)
            grad = torch.autograd.grad(loss, self.policy.parameters(), create_graph=True)
            grad = parameters_to_vector(grad)
            grad_F_inv_grad = torch.dot(grad, ng_grad_0)
            ng_grad_1 = torch.autograd.grad(grad_F_inv_grad, self.policy.parameters())
            ng_grad_1 = parameters_to_vector(ng_grad_1)

            ng_grad = ng_grad_0 - step_size * ng_grad_1

            ng_grad = parameters_to_vector(ng_grad)
            ng_grads.append(ng_grad.view(len(ng_grad), 1))

        return torch.mean(torch.stack(ng_grads, dim=1), dim=[1, 2])

    def step_test(self, episodes, max_kl=1e-3, cg_iters=10, cg_damping=1e-2,
             ls_max_steps=10, ls_backtrack_ratio=0.5):
        """Meta-optimization step (ie. update of the initial parameters), based
        on Trust Region Policy Optimization (TRPO, [4]).
        """
        grads = self.compute_ng_gradient_test(episodes)
        old_params = parameters_to_vector(self.policy.parameters())
        update_params = self.adam_step(old_params, grads)
        vector_to_parameters(update_params, self.policy.parameters())

    def sample_test(self, tasks, first_order=False):
        """Sample trajectories (before and after the update of the parameters)
        for all the tasks `tasks`.
        """
        episodes = []
        for task in tasks:
            self.sampler.reset_task(task)
            train_episodes = self.sampler.sample(self.policy,
                gamma=self.gamma, device=self.device)

            params, _, _ = self.adapt_ng_test(train_episodes, first_order=first_order)

            valid_episodes = self.sampler.sample(self.policy, params=params,
                gamma=self.gamma, device=self.device)
            episodes.append((train_episodes, valid_episodes))
        return episodes

    def adapt_ng_test(self, episodes, first_order=False, max_kl=1e-3, cg_iters=20, cg_damping=1e-2,
             ls_max_steps=10, ls_backtrack_ratio=0.5):
        """Adapt the parameters of the policy network to a new task, from
        sampled trajectories `episodes`, with a one-step natural gradient update.
        """
        # Fit the baseline to the training episodes
        self.baseline.fit(episodes)
        # Get the loss on the training episodes
        loss_lvc = self.inner_loss_lvc(episodes)
        # Get the new parameters after a one-step natural gradient update
        grads = torch.autograd.grad(loss_lvc, self.policy.parameters())
        # grads = parameters_to_vector(grads)

        step_size = 0.1
        params = OrderedDict()
        for (name, param), grad in zip(self.policy.named_parameters(), grads):
            params[name] = param - step_size*grad

        return params, step_size, grads