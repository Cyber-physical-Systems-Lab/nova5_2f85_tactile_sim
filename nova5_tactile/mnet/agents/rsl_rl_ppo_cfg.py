from isaaclab.utils.configclass import configclass
from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


@configclass
class PegInHolePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 32
    max_iterations = 300
    save_interval = 50
    experiment_name = 'mnet_peg_in_hole'
    actor = RslRlMLPModelCfg(hidden_dims=[256, 128, 64], activation='elu', obs_normalization=True,
                             distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=.6))
    critic = RslRlMLPModelCfg(hidden_dims=[256, 128, 64], activation='elu', obs_normalization=True)
    algorithm = RslRlPpoAlgorithmCfg(value_loss_coef=1., use_clipped_value_loss=True, clip_param=.2, entropy_coef=.002,
                                     num_learning_epochs=5, num_mini_batches=4, learning_rate=3e-4, schedule='adaptive',
                                     gamma=.99, lam=.95, desired_kl=.01, max_grad_norm=1.)
