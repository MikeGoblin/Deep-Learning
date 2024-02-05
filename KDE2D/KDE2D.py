'''
2D Kernel Density Estimation with PyTorch.
'''

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# configuration
config = {}
config['grid_size'] = (10, 100)
config['bandwidth_init'] = 1.0
config['batch_size'] = 128
config['timesteps'] = 16
config['n_points'] = 40

class KDE2D(nn.Module):
    def __init__(self):
        super(KDE2D, self).__init__()

        grid_size = config['grid_size']
        bandwidth_init = config['bandwidth_init']

        # Initialize the bandwidth as a learnable parameter
        self.bandwidth = nn.Parameter(torch.tensor(bandwidth_init))
        self.grid_size = grid_size
        self.x_grid = torch.linspace(-5, 5, grid_size[0])
        self.y_grid = torch.linspace(-5, 5, grid_size[1])
        self.X, self.Y = torch.meshgrid(self.x_grid, self.y_grid, indexing='ij')

    def forward(self, A):
        # A is expected to be of shape [Batchsize, Timestep, n_points, 2 (for x and y)]
        batch_size, timesteps, n_points, _ = A.shape
        # Initialize the KDE results tensor
        kde_results = torch.zeros(batch_size, timesteps, *self.grid_size, device=A.device)

        for i in range(batch_size):
            for t in range(timesteps):
                # Extract x and y values
                points = A[i, t, :, :]
                x = points[:, 0]
                y = points[:, 1]

                # Normalize the input data points
                x = (x - x.mean()) / x.std()
                y = (y - y.mean()) / y.std()

                # Calculate the KDE for each point in the grid
                kde_results[i, t] = self._calculate_kde(x, y)

        return kde_results

    def _calculate_kde(self, x, y):
        # Calculate the Gaussian kernel for each point in the grid
        kernel = (1 / (2 * torch.pi * self.bandwidth ** 2)) * torch.exp(
            -0.5 * (((self.X.unsqueeze(-1) - x) / self.bandwidth) ** 2 + ((self.Y.unsqueeze(-1) - y) / self.bandwidth) ** 2))
        # Sum the kernels for all data points
        density = kernel.sum(dim=-1)
        return density
    

def visualize_kde(kde_results, grid_size, batch_index=0, time_step=0):
    """
    Visualizes the KDE results for a given batch index and time step.
    """
    # Select the density estimation for the given batch index and time step
    density = kde_results[batch_index, time_step].detach().numpy()
    
    # Create a meshgrid for plotting
    x_grid = torch.linspace(-4, 4, grid_size[0])
    y_grid = torch.linspace(-4, 4, grid_size[1])
    X, Y = torch.meshgrid(x_grid, y_grid, indexing='ij')
    
    # Plot the results
    plt.figure(figsize=(8, 6))
    plt.pcolormesh(X, Y, density, shading='auto')
    plt.colorbar()  # Show color scale
    plt.title(f'2D KDE Visualization (Batch {batch_index}, Time Step {time_step})')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.show()


def Example(batch_size, timesteps, n_points):
    # n_points denotes the number of x, y points per timestep

    # Random input tensor representing [Batchsize, Timestep, n_points, 2 (for x and y)]
    A = torch.randn(batch_size, timesteps, n_points, 2)

    # Initialize the KDE module
    kde_module = KDE2D()

    kde_results = kde_module(A)

    print(kde_results.shape) 

    # Visualize the KDE for the first batch and first timestep
    visualize_kde(kde_results, config['grid_size'], batch_index=0, time_step=0)


if __name__ == '__main__':
    batch_size = config['batch_size']
    timesteps = config['timesteps']
    n_points = config['n_points']
    Example(batch_size, timesteps, n_points)