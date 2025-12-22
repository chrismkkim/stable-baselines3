'''
Save following variables when braked in
    a2c.py
        class A2C
            def train()
                actions = rollout_data.actions
'''
dirpath = '/Users/kimchm/Documents/RL/temp/'
th.save(self.policy.mlp_extractor.policy_net[0].weight, dirpath + 'policy0_a2c.pt')
th.save(self.policy.mlp_extractor.policy_net[2].weight, dirpath + 'policy2_a2c.pt')
th.save(self.policy.mlp_extractor.value_net[0].weight, dirpath + 'value0_a2c.pt')
th.save(self.policy.mlp_extractor.value_net[2].weight, dirpath + 'value2_a2c.pt')


'''
Save following variables when braked in
    dopa.py
        class Dopa
            def rl_dopa()
                loss_rl = self.compute_rlloss_using_dopa_interpolated(rollout_data)
'''
dirpath = '/Users/kimchm/Documents/RL/temp/'
th.save(self.policy.mlp_extractor.policy_net[0].weight, dirpath + 'policy0_tdnet.pt')
th.save(self.policy.mlp_extractor.policy_net[2].weight, dirpath + 'policy2_tdnet.pt')
th.save(self.policy.mlp_extractor.value_net[0].weight, dirpath + 'value0_tdnet.pt')
th.save(self.policy.mlp_extractor.value_net[2].weight, dirpath + 'value2_tdnet.pt')

policy0_a2c   = th.load(dirpath + 'policy0_a2c.pt', weights_only=False)
policy2_a2c   = th.load(dirpath + 'policy2_a2c.pt', weights_only=False)
value0_a2c    = th.load(dirpath + 'value0_a2c.pt', weights_only=False)
value2_a2c    = th.load(dirpath + 'value2_a2c.pt', weights_only=False)
policy0_tdnet = th.load(dirpath + 'policy0_tdnet.pt', weights_only=False)
policy2_tdnet = th.load(dirpath + 'policy2_tdnet.pt', weights_only=False)
value0_tdnet  = th.load(dirpath + 'value0_tdnet.pt', weights_only=False)
value2_tdnet  = th.load(dirpath + 'value2_tdnet.pt', weights_only=False)

'''
Compare the weights of policy_net and value_net
'''
print(th.all(policy0_a2c == policy0_tdnet))
print('\n',th.all(policy2_a2c == policy2_tdnet))
print('\n',th.all(value0_a2c == value0_tdnet))
print('\n',th.all(value2_a2c == value2_tdnet))

