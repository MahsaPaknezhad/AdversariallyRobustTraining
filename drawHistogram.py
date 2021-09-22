from MnistDataHandler3 import DataHandler
from MnistNoiseDataGenerator1 import NoiseDataGenerator
import numpy as np
import matplotlib.pyplot as plt

class DrawHistogram:

    def draw_histogram(self):
        histogram_values = []
        noise_generator_mean = 0
        single_noise_mean = 0
        single_noise_var = 0.05
        num_samples_per_class = 500
        train_val_ratio = 0.8
        clip = False
        num_bins = 20
        noise_mode = 'gaussian'
        noise_data_generator = NoiseDataGenerator(single_noise_mean, single_noise_var, clip)
        data_handler = DataHandler(noise_data_generator, noise_mode)
        [x_train_tensor, y_train_tensor, train_shape, x_test_tensor, y_test_tensor, test_shape,
         x_neighbors_tensor, y_neighbors_tensor, neighbors_shape] = data_handler.prepareDataset(num_samples_per_class,
                                                                         100, train_val_ratio)
        for i in np.arange(0, train_shape[0]):
            for j in np.arange(0, train_shape[0]):
                if i!=j:
                    image1 = x_train_tensor[i,...]
                    image2 = x_train_tensor[j,...]
                    diff = (image1 - image2)**2
                    diff = diff.flatten()
                    hist_value = np.sqrt(np.sum(diff))/(train_shape[1] * train_shape[2])
                    histogram_values.append(hist_value)

        # An "interface" to matplotlib.axes.Axes.hist() method
        histogram_values_np = np.asarray(histogram_values)
        mu = np.mean(histogram_values_np)
        var = np.var(histogram_values_np)
        print(r'$\mu$=%f, var=%f'%(mu, var))
        n, bins, patches = plt.hist(x=histogram_values_np, bins='auto', color='#0504aa',
                                    alpha=0.7, rwidth=0.85)
        plt.grid(axis='y', alpha=0.75)
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.title('L2 Norm Histogram')
        plt.text(0.02, 7000, r'$\mu$=%f, var=%f'%(mu, var))
        maxfreq = n.max()
        # Set a clean upper y-axis limit.
        plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
        plt.show()


if __name__ == '__main__':
    HistFunc = DrawHistogram()
    HistFunc.draw_histogram()


