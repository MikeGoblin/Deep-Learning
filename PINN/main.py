import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.fc1 = nn.Linear(2, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)
    
    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        x = self.fc4(x)
        return x
    

def loss_function(x, y, model, criterion=nn.MSELoss()):
    x.requires_grad_(True)
    y.requires_grad_(True)
    
    u = model(torch.cat([x, y], dim=1))
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True, retain_graph=True)[0]
    u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(u_y), create_graph=True, retain_graph=True)[0]
    u_yyy = torch.autograd.grad(u_yy, y, grad_outputs=torch.ones_like(u_yy), create_graph=True, retain_graph=True)[0]
    u_yyyy = torch.autograd.grad(u_yyy, y, grad_outputs=torch.ones_like(u_yyy), create_graph=True)[0]
    
    # PDE loss
    pde_residual = u_xx - u_yyyy - (2 - x**2) * torch.exp(-y)
    pde_loss = criterion(pde_residual, torch.zeros_like(pde_residual))
    
    # Boundary conditions
    # u_yy(x,0) = x^2
    y_0 = torch.zeros_like(x)
    y_0.requires_grad_(True)
    u_y_0 = model(torch.cat([x, y_0], dim=1))
    u_y_y_0 = torch.autograd.grad(u_y_0, y_0, grad_outputs=torch.ones_like(u_y_0), create_graph=True, retain_graph=True)[0]
    u_yy_x_0 = torch.autograd.grad(u_y_y_0, y_0, grad_outputs=torch.ones_like(u_y_y_0), create_graph=True)[0]
    bc1_loss = criterion(u_yy_x_0, x**2)
    
    # u_yy(x,1) = x^2 / e
    y_1 = torch.ones_like(x)
    y_1.requires_grad_(True)
    u_y_1 = model(torch.cat([x, y_1], dim=1))
    u_y_y_1 = torch.autograd.grad(u_y_1, y_1, grad_outputs=torch.ones_like(u_y_1), create_graph=True, retain_graph=True)[0]
    u_yy_x_1 = torch.autograd.grad(u_y_y_1, y_1, grad_outputs=torch.ones_like(u_y_y_1), create_graph=True)[0]
    bc2_loss = criterion(u_yy_x_1, x**2 / torch.exp(torch.ones_like(x)))
    
    # u(x,0) = x^2
    u_x_0 = model(torch.cat([x, y_0], dim=1))
    bc3_loss = criterion(u_x_0, x**2)
    
    # u(x,1) = x^2 / e
    u_x_1 = model(torch.cat([x, y_1], dim=1))
    bc4_loss = criterion(u_x_1, x**2 / torch.exp(torch.ones_like(x)))
    
    # u(0,y) = 0
    x_0 = torch.zeros_like(y)
    u_0_y = model(torch.cat([x_0, y], dim=1))
    bc5_loss = criterion(u_0_y, torch.zeros_like(u_0_y))
    
    # u(1,y) = e^(-y)
    x_1 = torch.ones_like(y)
    u_1_y = model(torch.cat([x_1, y], dim=1))
    bc6_loss = criterion(u_1_y, torch.exp(-y))
    
    # Total loss
    total_loss = pde_loss + bc1_loss + bc2_loss + bc3_loss + bc4_loss + bc5_loss + bc6_loss
    
    return total_loss


if __name__ == '__main__':
    model = PINN()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    n = 256
    # Training loop
    for epoch in range(10000):
        optimizer.zero_grad()
        x = torch.rand((n, 1), requires_grad=True)
        y = torch.rand((n, 1), requires_grad=True)
        loss = loss_function(x, y, model)
        loss.backward()
        optimizer.step()
        if epoch % 1000 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')

    # Test and Drawing
    h = 100
    x_test = torch.linspace(0, 1, h).reshape(-1, 1)
    y_test = torch.linspace(0, 1, h).reshape(-1, 1)
    xx, yy = torch.meshgrid(x_test.squeeze(), y_test.squeeze())
    test_input = torch.cat((xx.reshape(-1, 1), yy.reshape(-1, 1)), dim=1)
    
    u_pred = model(test_input).detach().numpy().reshape(h, h)
    u_real = (xx.numpy() ** 2) * np.exp(-yy.numpy())
    u_error = np.abs(u_pred - u_real)
    max_abs_error = np.max(u_error)
    print("Max abs error is: ", max_abs_error)

    # PINN Prediction Figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(xx.numpy(), yy.numpy(), u_pred, cmap='viridis')
    ax.set_title('PINN Prediction')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('u')
    plt.show()

    # Real Solution Figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(xx.numpy(), yy.numpy(), u_real, cmap='viridis')
    ax.set_title('Real Solution')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('u')
    plt.show()

    # Error Figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(xx.numpy(), yy.numpy(), u_error, cmap='viridis')
    ax.set_title('Error')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('Error')
    plt.show()
    