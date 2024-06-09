# A Practice Example for PINN

## PDE

$$\frac{\partial^2 u}{\partial x^2} - \frac{\partial^4 u}{\partial y^4} = (2 - x^2) e^{-y}$$

## Boundary Conditions

The following boundary conditions are considered:

$$u_{yy}(x,0) = x^2$$

$$u_{yy}(x,1) = \frac{x^2}{e}$$

$$u(x,0) = x^2$$

$$u(x,1) = \frac{x^2}{e}$$

$$u(0,y) = 0$$

$$u(1,y) = e^{-y}$$

## Results

Figure of ***PINN Predictions*** is shown below.

![image](https://github.com/MikeGoblin/Deep-Learning/blob/main/PINN/Figures/PINN_Prediction.png?raw=true)

Figure of ***Real Solution*** is shown below.

![image](https://github.com/MikeGoblin/Deep-Learning/blob/main/PINN/Figures/Real_Solution.png?raw=true)

And the figure of ***Error*** is shown below.

![image](https://github.com/MikeGoblin/Deep-Learning/blob/main/PINN/Figures/Error.png?raw=true)
