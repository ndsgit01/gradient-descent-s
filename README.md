# gradient-descents

Contents:
-
- gradient_descent.py : An 'over-engineered(modular)' OOP implementation of gradient descent algorithms and its popular variants in python (+ numpy)
- Visualization.ipynb : Observing the behaviour of above implemented methods with animated visualizations in python (+ matplotlib)

Algorithms implemented:
- 
- Vanilla Gradient Descent <br>
  ![GradientDescent](https://github.com/ndsgit01/gradient-descent-s/assets/51270897/fb064cde-1c4d-4da1-a827-1209421c5454) <br>
- Momentum Gradient Descent (Inherits Vanilla Gradient Descent) <br>
  ![MomentumGradientDescent](https://github.com/ndsgit01/gradient-descent-s/assets/51270897/d11acd3f-b632-4b82-9f95-169232ded6a2) <br>
- Nesterov Gradient Descent (Inherits Momentum Gradient Descent) <br>
  ![NesterovGradientDescent1D](https://github.com/ndsgit01/gradient-descent-s/assets/51270897/1ead840d-1d62-48f3-9030-a959c0d81496) <br>
  ![NesterovGradientDescent2D](https://github.com/ndsgit01/gradient-descent-s/assets/51270897/9c00c0a1-dff2-4774-a4a3-2313eec86b38) <br>
- AdaGrad Gradient Descent (Inherits Vanilla Gradient Descent) <br>
- RMSProp Gradient Descent (Inherits AdaGrad Gradient Descent) <br>
  ![RMSPropGradientDescent](https://github.com/ndsgit01/gradient-descent-s/assets/51270897/42facc50-9fb3-4ad2-9a5e-fa4bd5380754) <br>
- Adam Gradient Descent (Inherits Vanilla Gradient Descent) {Note: Avoided the 'logical' multiple inheritance from Momentum & RMSProp, because it looked messy} <br>
  ![AdamGradientDescent](https://github.com/ndsgit01/gradient-descent-s/assets/51270897/3e68f12a-8c4f-4fa7-94ac-8af759aa81a3) <br>

Future plans:
-
- The implemented algorithms should also work with stochastic and batch gradient descent - have to confirm the same by testing whether a gradient function that yields the 'computed'(based on stochastic / batch version of gradient descent) gradient performs correctly.
- Include a cleaner code for dynamic update of ax.quiver in animation function once https://github.com/matplotlib/matplotlib/pull/22407 is available.
- Include variants of Adam Gradient Descent (or other interesting Gradient Descent algorithms)
  
