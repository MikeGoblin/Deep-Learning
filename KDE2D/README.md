# 2D Kernel Density Estimation with PyTorch

Kernel Density Estimation (KDE) is a powerful non-parametric method to estimate the probability density function of a random variable. In the realm of data analysis and machine learning, understanding the underlying distribution of data can be crucial for various tasks such as anomaly detection, data smoothing, and feature extraction. While KDE is widely used for one-dimensional data, its extension to two or more dimensions opens up a plethora of opportunities for analyzing complex datasets.

## Key Features of the `KDE2D` Demo:

- **2D Support**: The demo is tailored to perform KDE on two-dimensional data, making it suitable for applications such as image processing, spatial data analysis, and any scenario where data can be naturally represented in two dimensions.
- **Learnable Bandwidth**: The bandwidth parameter, which dictates the smoothness of the estimated density, is implemented as a learnable PyTorch parameter. This allows for the optimization of the bandwidth during model training, using backpropagation and gradient descent.
- **Batch and Time-step Processing**: To cater to the needs of temporal data or batch processing, the module is designed to handle inputs with additional dimensions for batch size and time steps, making it well-suited for time-series analysis and batch training scenarios.

* **Visualization Functionality** : Includes a `visualize_kde` function to easily plot the results of the KDE, providing immediate visual feedback. This is essential for understanding the distribution of data points and the smoothness of the density estimation.
