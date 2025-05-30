import numpy as np
np.random.seed(666)

N = 3010
n = N//10

x = np.random.choice(N, size=n*2, replace=False)
np.savetxt('idx_test.dat', np.concatenate((x[:n], x[:n]+N)), fmt='%d')
np.savetxt('idx_val.dat', np.concatenate((x[n:], x[n:]+N)), fmt='%d')
