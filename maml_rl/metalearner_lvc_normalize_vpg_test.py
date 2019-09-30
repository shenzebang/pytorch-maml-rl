import torch
from torch.nn.utils.convert_parameters import (vector_to_parameters,
                                               parameters_to_vector)
from torch.distributions.kl import kl_divergence

from maml_rl.utils.torch_utils import (weighted_mean, detach_distribution,
                                       weighted_normalize)
from maml_rl.utils.optimization import conjugate_gradient

class MetaLearnerLVCNormalizeVPGTest(object):
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
                 fast_lr=0.5, tau=1.0, device='cpu'):
        self.sampler = sampler
        self.policy = policy
        self.baseline = baseline
        self.gamma = gamma
        self.fast_lr = fast_lr
        self.tau = tau
        self.to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), 5e-4, eps=1e-5)
        # for adam optimizer
        self.v = 0
        self.sqr = 0
        self.t = 0

    def inner_loss(self, episodes, params=None):
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

    def adapt(self, episodes, first_order=False):
        """Adapt the parameters of the policy network to a new task, from
        sampled trajectories `episodes`, with a one-step gradient update [1].
        """
        # Fit the baseline to the training episodes
        self.baseline.fit(episodes)
        # Get the loss on the training episodes
        loss = self.inner_loss(episodes)
        # Get the new parameters after a one-step gradient update
        params = self.policy.update_params(loss, step_size=self.fast_lr,
            first_order=first_order)

        return params

    def adapt_first_order(self, episodes, first_order=False):
        """Adapt the parameters of the policy network to a new task, from
        sampled trajectories `episodes`, with a one-step gradient update [1].
        """
        # Fit the baseline to the training episodes
        self.baseline.fit(episodes)
        # Get the loss on the training episodes
        loss = self.inner_loss(episodes)
        # Get the new parameters after a one-step gradient update
        params = self.policy.update_params(loss, step_size=self.fast_lr,
            first_order=True)

        return params

    def sample(self, tasks, first_order=False):
        """Sample trajectories (before and after the update of the parameters) 
        for all the tasks `tasks`.
        """
        episodes = []
        for task in tasks:
            self.sampler.reset_task(task)
            train_episodes = self.sampler.sample(self.policy,
                gamma=self.gamma, device=self.device)

            params = self.adapt(train_episodes, first_order=first_order)

            valid_episodes = self.sampler.sample(self.policy, params=params,
                gamma=self.gamma, device=self.device)
            episodes.append((train_episodes, valid_episodes))
        return episodes

    def kl_divergence(self, episodes, old_pis=None):
        kls = []
        if old_pis is None:
            old_pis = [None] * len(episodes)

        for (train_episodes, valid_episodes), old_pi in zip(episodes, old_pis):
            params = self.adapt(train_episodes)
            pi = self.policy(valid_episodes.observations, params=params)

            if old_pi is None:
                old_pi = detach_distribution(pi)

            mask = valid_episodes.mask
            if valid_episodes.actions.dim() > 2:
                mask = mask.unsqueeze(2)
            kl = weighted_mean(kl_divergence(old_pi, pi), dim=0, weights=mask)
            kls.append(kl)

        return torch.mean(torch.stack(kls, dim=0))

    def hessian_vector_product(self, episodes, damping=1e-2):
        """Hessian-vector product, based on the Perlmutter method."""
        def _product(vector):
            kl = self.kl_divergence(episodes)
            grads = torch.autograd.grad(kl, self.policy.parameters(),
                create_graph=True)
            flat_grad_kl = parameters_to_vector(grads)

            grad_kl_v = torch.dot(flat_grad_kl, vector)
            grad2s = torch.autograd.grad(grad_kl_v, self.policy.parameters())
            flat_grad2_kl = parameters_to_vector(grad2s)

            return flat_grad2_kl + damping * vector
        return _product

    def surrogate_loss(self, episodes, old_pis=None):
        losses, kls, pis = [], [], []
        if old_pis is None:
            old_pis = [None] * len(episodes)

        for (train_episodes, valid_episodes), old_pi in zip(episodes, old_pis):
            params = self.adapt(train_episodes)
            with torch.set_grad_enabled(old_pi is None):
                pi = self.policy(valid_episodes.observations, params=params)
                pis.append(detach_distribution(pi))

                if old_pi is None:
                    old_pi = detach_distribution(pi)

                values = self.baseline(valid_episodes)
                advantages = valid_episodes.gae(values, tau=self.tau)
                advantages = weighted_normalize(advantages,
                    weights=valid_episodes.mask)

                log_ratio = (pi.log_prob(valid_episodes.actions)
                    - old_pi.log_prob(valid_episodes.actions))
                if log_ratio.dim() > 2:
                    log_ratio = torch.sum(log_ratio, dim=2)
                ratio = torch.exp(log_ratio)

                loss = -weighted_mean(ratio * advantages, dim=0,
                    weights=valid_episodes.mask)
                losses.append(loss)

                mask = valid_episodes.mask
                if valid_episodes.actions.dim() > 2:
                    mask = mask.unsqueeze(2)
                kl = weighted_mean(kl_divergence(old_pi, pi), dim=0,
                    weights=mask)
                kls.append(kl)

        return (torch.mean(torch.stack(losses, dim=0)),
                torch.mean(torch.stack(kls, dim=0)), pis)

    def step(self, episodes, max_kl=1e-3, cg_iters=10, cg_damping=1e-2,
             ls_max_steps=10, ls_backtrack_ratio=0.5):
        """Meta-optimization step (ie. update of the initial parameters), based 
        on Trust Region Policy Optimization (TRPO, [4]).
        """
        old_loss, _, _ = self.surrogate_loss(episodes)
        grads = torch.autograd.grad(old_loss, self.policy.parameters())
        grads = parameters_to_vector(grads)

        grads2 = self.compute_grad(episodes)
        print(torch.norm(grads - grads2))

        old_params = parameters_to_vector(self.policy.parameters())
        update_params = self.adam_step(old_params, grads2)
        vector_to_parameters(update_params, self.policy.parameters())
        # self.optimizer.zero_grad()
        # old_loss.backward()
        # self.optimizer.step()

    def to(self, device, **kwargs):
        self.policy.to(device, **kwargs)
        self.baseline.to(device, **kwargs)
        self.device = device

    def compute_grad(self, episodes):
        ng_grads = []
        for train_episodes, valid_episodes in episodes:
            params_adapt = self.adapt_first_order(train_episodes)

            # self.baseline.fit(valid_episodes)
            loss = self.inner_loss(valid_episodes, params=params_adapt)
            ng_grad_0 = torch.autograd.grad(loss, self.policy.parameters())  # no create graph
            ng_grad_0 = parameters_to_vector(ng_grad_0)

            self.baseline.fit(train_episodes)
            loss = self.inner_loss(train_episodes)
            grad = torch.autograd.grad(loss, self.policy.parameters(), create_graph=True)
            grad = parameters_to_vector(grad)
            grad_ng_grad_0 = torch.dot(grad, ng_grad_0)
            ng_grad_1 = torch.autograd.grad(grad_ng_grad_0, self.policy.parameters())
            ng_grad_1 = parameters_to_vector(ng_grad_1)

            ng_grad = ng_grad_0 - 0.1 * ng_grad_1

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