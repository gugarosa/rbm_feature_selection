import learnergy.visual.convergence as c

# Defines the metrics to be plotted (lists)
fsrbm = [50, 40, 30, 20, 10]
fsrbm_mask = [51.2, 42.4, 34.1, 21.3, 11.8]
rbm = [48, 37, 28, 17, 8]
rbm_mask = [50.3, 37.58, 31.23, 19.74, 9.98]

# Plots the desired metrics
c.plot(fsrbm, fsrbm_mask, rbm, rbm_mask, labels=['FSRBM', 'FSRBM-mask', 'RBM', 'RBM-mask'],
       title='MNIST: Average Reconstruction Error ($25$ runs)',
       subtitle='Set: Testing', xlabel='Snapshot', ylabel='MSE')
