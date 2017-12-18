import numpy as np
import matplotlib.pyplot as plt

def two_moons(num_samples,moon_radius,moon_var):
    X = np.zeros((num_samples,2))
    for i in range(num_samples//2):
        r = moon_radius + 4 * (i-1)/num_samples
        t = (i - 1) * 3/num_samples * np.pi
        X[i,0] = r*np.cos(t)
        X[i,1] = r*np.sin(t)
        X[i + num_samples//2, 0] = r*np.cos(t + np.pi)
        X[i + num_samples//2, 1] = r*np.sin(t + np.pi)

    X += np.sqrt(moon_var) * np.random.randn(num_samples, 2)
    Y = np.concatenate((np.zeros(num_samples//2, dtype=int),np.ones(num_samples//2, dtype=int)))

    return X,Y

if __name__ == '__main__':
    X,Y = two_moons(1000,1,1e-2)
    print(X.shape,Y.shape)
    plt.scatter(X[:,0],X[:,1],c=Y)
    plt.show()

