import numpy as np
import torch
from marl.rlcore.algo import JointPPO, PPO
from marl.rlagent import Neo
from marl.mpnn import MPNN

def master(args):
    policy1 = MPNN(input_size=args.pol_obs_dim,num_agents=1,num_entities=0,action_space=args.action_space,
                                mask_dist=args.mask_dist).to(args.device)
    agents=[Neo(args,policy1,(args.pol_obs_dim,),args.action_space)]
    masterr=Learner(args,agents,[policy1])
    return masterr
    


class Learner (object):
    def __init__(self, args, agents, policies):
        self.args=args
        self.device = args.device
        self.policies=[x for x in policies if x is not None]
        self.all_agents = [agent for agent in agents]
        #self.env=env
        self.trainers = [JointPPO(policy, args.clip_param, args.ppo_epoch, args.num_mini_batch, args.value_loss_coef, ##############check
                                       args.entropy_coef, lr=args.lr, max_grad_norm=args.max_grad_norm,
                                       use_clipped_value_loss=args.clipped_value_loss) for policy in self.policies]
    
    def act(self,step):
        actions=[]
        for policy in self.policies:
            #print([agent.rollouts.obs for agent in self.all_agents if agent.rollouts.obs[-1] is not None])
            all_obs = torch.cat([agent.rollouts.obs[step] for agent in self.all_agents])
            all_hidden = torch.cat([agent.rollouts.recurrent_hidden_states[step] for agent in self.all_agents])
            all_masks = torch.cat([agent.rollouts.masks[step] for agent in self.all_agents])

            props = policy.act(all_obs, all_hidden, all_masks, deterministic=False)

            n = len(self.all_agents)
            _, all_action, _,_ = [torch.chunk(x, n) for x in props]

            for i in range(n):
                actions.append(all_action[i].cpu().numpy())
        
        return actions
    def update(self):
        return_vals=[]
        for i, trainer in enumerate(self.trainers):
            rollouts_list = [agent.rollouts for agent in self.all_agents]
            vals = trainer.update(rollouts_list)
            return_vals.append([np.array(vals)]*len(rollouts_list))
        
        return np.stack([x for v in return_vals for x in v]).reshape(-1,3)

    def update_rollout(self, obs, reward, masks):
        obs_t = torch.from_numpy(obs).float().to(self.device)
        
        for i, agent in enumerate(self.all_agents):
            agent_obs = obs_t
            
            agent.update_rollout(agent_obs, reward, masks)