import torch
import math
import numpy as np

# mu, std = 2, 4
#
# q = torch.distributions.Normal(mu, std)
# p = torch.distributions.Normal(0, 1)
#
# kl = []
# for _ in range(100000):
#     z = q.rsample()
#     log_q_z = q.log_prob(z)
#     log_p_z = p.log_prob(z)
#     kl.append(log_q_z - log_p_z)
#
# kl = np.mean(kl)
#
# print("Est. KL", kl)
#
# formula_kl = -1 / 2 * (1 + math.log(std ** 2) - mu ** 2 - std ** 2)
#
# print(formula_kl)


x = [[] for _ in range(10)]

y = torch.stack()

