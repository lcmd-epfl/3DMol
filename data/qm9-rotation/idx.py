import numpy as np

for i in range(20):

    if i>=10:
        seed = 666 + (i - 10)
    else:
        seed = 123 + i

    N = 205878//2
    n = N//10

    x = np.random.choice(N, size=n*2, replace=False)
    np.savetxt(f'splits/idx_test.{i}.dat', np.concatenate((x[:n], x[:n]+N)), fmt='%d')
    np.savetxt(f'splits/idx_val.{i}.dat', np.concatenate((x[n:], x[n:]+N)), fmt='%d')
