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

![image](https://user-images.githubusercontent.com/62187390/136612467-68324140-f9e9-4587-8587-60086254120d.png)

Figure of ***Real Solution*** is shown below.

![image](https://user-images.githubusercontent.com/62187390/136612531-7d971589-4465-4702-8637-183492188222.png)

And the figure of ***Error*** is shown below.

![image](https://user-images.githubusercontent.com/62187390/136612531-7d971589-4465-4702-8637-183492188222.png)
