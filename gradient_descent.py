import numpy as np



class GradientDescent:
  """
  Implements the vanilla gradient descent algorithm.

  Attributes
  ----------
  f : callable
      Evaluates the target function
  grad_f : callable
           Evaluates gradient of the target function f
  eta: float
       Constant learning rate
  history: list
           Sequence of function parameters w's encountered in the gradient
           descent
  """

  def __init__(self, f, grad_f, eta=1e-2):
    self.f = f
    self.grad_f = grad_f
    self.eta = eta
    self.history = []

  def get_lr(self):
    """
    Returns learning rate for the current iteration. In the vanilla version the
    lr is the constant eta.

    Returns
    -------
    out: float
         Learning rate for the current iteration
    """
    return self.eta

  def get_update(self):
    """
    Returns update vector for the current iteration.
    - learning_rate * grad_f(w_old)
    """
    w_old = self.history[-1]
    grad = self.grad_f(w_old)
    update = - (self.get_lr() * grad)
    return update

  def step(self):
    """
    Makes one step of the gradient descent.
    w_new = w_old + update vector
    """
    w_old = self.history[-1]
    w_new = w_old + self.get_update()
    self.history.append(w_new)

  def stop_criteria(self, eps):
    """
    Test if the stopping criterion is statisfied.
    ||w_new - w_old|| < eps

    Parameters
    ----------
    eps: tolerance

    Returns
    -------
    out: bool
         status of stopping criteria
    """
    return np.linalg.norm(self.history[-2] - self.history[-1]) < eps

  def iterate(self, w_0, max_iter=int(1e4), eps=1e-4):
    """
    Performs gradient descent starting from w_0 for max_iter iterations
    On each iteration it
    - makes step using `self.step`
    - check stopping criterion using tolerance eps.
    Returns local minimum found by gradient descent after iteration breaks

    Parameters
    ----------
    w_0: array_like
         initial weight parameter
    max_iter: int
              Maximum number of iterations in the gradient descent
    eps: float
         Tolerance level used in stopping criteria

    Returns
    -------
    out: tuple
         Local minimum found by gradient descent and the value of target
         function at local minimum
    """
    self.history.append(w_0)
    for _ in range(max_iter):
      self.step()
      if self.stop_criteria(eps):
        break
    return self.history[-1], self.f(self.history[-1]), len(self.history)


class MomentumGradientDescent(GradientDescent):

  def __init__(self, f, grad_f, eta=1e-2, mu=0.9):
    """
    Implements the moment method gradient descent algorithm.

    Attributes
    ----------
    f : callable
        Evaluates the target function
    grad_f : callable
             Evaluates gradient of the target function f
    eta: float
         Constant learning rate
    history: list
             Sequence of function parameters w's encountered in the gradient
             descent
    mu: float
        Constant momentum decay factor
    momentum_history: list
                      Sequence of gradient momentum contributions encountered
                      in the gradient descent
    """
    super().__init__(f, grad_f, eta)
    self.momentum_history = []
    self.mu = mu

  def get_mdf(self):
    """
    Returns momentum decay factor for the current iteration. In this version the
    mdf is the constant mu.

    Returns
    -------
    out: float
         Momentum decay factor for the current iteration
    """
    return self.mu

  def get_update(self):
    """
    Returns update vector for the current iteration.
    \mu * momentum_old - \eta * grad_f(w_old)
    """
    w_old = self.history[-1]
    momentum_old = self.momentum_history[-1]
    grad = self.grad_f(w_old)
    momentum_new = ((self.get_mdf() * momentum_old)
                  - (self.get_lr() * grad))
    self.momentum_history.append(momentum_new)
    return momentum_new

  def iterate(self, w_0, max_iter=int(1e4), eps=1e-4):
    """
    Performs gradient descent starting from w_0 for max_iter iterations
    On each iteration it
    - makes step using `self.step`
    - check stopping criterion using tolerance eps.
    Returns local minimum found by gradient descent after iteration breaks

    Parameters
    ----------
    w_0: array_like
         initial weight parameter
    max_iter: int
              Maximum number of iterations in the gradient descent
    eps: float
         Tolerance level used in stopping criteria

    Returns
    -------
    out: tuple
         Local minimum found by gradient descent and the value of target
         function at local minimum
    """
    self.momentum_history.append(np.zeros_like(w_0))
    return super().iterate( w_0, max_iter, eps)


class NesterovGradientDescent(MomentumGradientDescent):

  def __init__(self, f, grad_f, lr=1e-2, mu=0.9):
    """
    Implements the nesterov accelerated gradient descent algorithm.

    Attributes
    ----------
    f : callable
        Evaluates the target function
    grad_f : callable
             Evaluates gradient of the target function f
    eta: float
         Constant learning rate
    history: list
             Sequence of function parameters w's encountered in the gradient
             descent
    mu: float
        Constant momentum decay factor
    momentum_history: list
                      Sequence of gradient momentum contributions encountered
                      in the gradient descent
    """
    super().__init__(f, grad_f, lr, mu)

  def get_update(self):
    """
    Returns update vector for the current iteration.
    w_lookahead = w_old + \mu * momentum_old
    momentum_new = \mu * momentum_old - \eta * grad_f(w_old)
    """
    w_old = self.history[-1]
    momentum_old = self.momentum_history[-1]
    w_lookahead = w_old + self.get_mdf() * momentum_old
    grad = self.grad_f(w_lookahead)
    momentum_new = ((self.get_mdf() * momentum_old)
                  - (self.get_lr() * grad))
    self.momentum_history.append(momentum_new)
    return momentum_new


class AdaGradGradientDescent(GradientDescent):

  def __init__(self, f, grad_f, eta=1e-2):
    """
    Implements the AdaGrad gradient descent algorithm.

    Attributes
    ----------
    f : callable
        Evaluates the target function
    grad_f : callable
             Evaluates gradient of the target function f
    eta: float
         Constant learning rate
    history: list
             Sequence of function parameters w's encountered in the gradient
             descent
    grad_square_sum_history: list
                         Sequence of sum of squares of gradient encountered in
                         the gradient descent
    """
    super().__init__(f, grad_f, eta)
    self.grad_square_sum_history = []

  def get_lr(self):
    """
    Returns learning rate for the current iteration i. In the adagrad version
    the  learning rate is adaptive for each feature w[k].
    \eta / sqrt(\sum{1}{i} (grad_f(w)[k])**2)

    Returns
    -------
    out: float
         Learning rate for the current iteration
    """
    w_old = self.history[-1]
    old_sum = self.grad_square_sum_history[-1]
    grad = self.grad_f(w_old)
    new_sum = old_sum + grad**2
    self.grad_square_sum_history.append(new_sum)
    return self.eta / ((new_sum)**0.5 + 1e-8)

  def iterate(self, w_0, max_iter=int(1e4), eps=1e-4):
    """
    Performs gradient descent starting from w_0 for max_iter iterations
    On each iteration it
    - makes step using `self.step`
    - check stopping criterion using tolerance eps.
    Returns local minimum found by gradient descent after iteration breaks

    Parameters
    ----------
    w_0: array_like
         initial weight parameter
    max_iter: int
              Maximum number of iterations in the gradient descent
    eps: float
         Tolerance level used in stopping criteria

    Returns
    -------
    out: tuple
         Local minimum found by gradient descent and the value of target
         function at local minimum
    """
    self.grad_square_sum_history.append(np.zeros_like(w_0))
    return super().iterate(w_0, max_iter, eps)


class RMSPropGradientDescent(AdaGradGradientDescent):

  def __init__(self, f, grad_f, eta=1e-2, beta=0.9):
    """
    Implements the RMSProp gradient descent algorithm.

    Attributes
    ----------
    f : callable
        Evaluates the target function
    grad_f : callable
             Evaluates gradient of the target function f
    eta: float
         Constant learning rate
    history: list
             Sequence of function parameters w's encountered in the gradient
             descent
    beta: float
          Constant gradient square sum decay factor
    grad_square_sum_history: list
                             Sequence of decaying sum of squares of gradient
                             encountered in the gradient descent
    """
    super().__init__(f, grad_f, eta)
    self.beta = beta

  def get_gssdf(self):
    """
    Returns gradient square sum decay factor for the current iteration.
    In this version the gssdf is the constant beta.

    Returns
    -------
    out: float
         Gradient square sum decay factor for the current iteration
    """
    return self.beta

  def get_lr(self):
    """
    Returns learning rate for the current iteration i. In the adagrad version
    the  learning rate is adaptive for each feature w[k].
    \eta / sqrt(\sum{1}{i} \beta**i(grad_f(w)[k])**2)

    Returns
    -------
    out: float
         Learning rate for the current iteration
    """
    w_old = self.history[-1]
    decay_sum_old = self.grad_square_sum_history[-1]
    grad = self.grad_f(w_old)
    decay_sum_new = ((self.get_gssdf() * decay_sum_old)
                     + ((1 - self.get_gssdf()) * grad**2))
    self.grad_square_sum_history.append(decay_sum_new)
    return self.eta / ((decay_sum_new)**0.5 + 1e-8)


class AdamGradientDescent(GradientDescent):

  def __init__(self, f, grad_f, eta=1e-2, mu=0.9, beta=0.999):
    """
    Implements the Adam gradient descent algorithm.

    Attributes
    ----------
    f : callable
        Evaluates the target function
    grad_f : callable
             Evaluates gradient of the target function f
    eta: float
         Constant learning rate
    history: list
             Sequence of function parameters w's encountered in the gradient
             descent
    mu: float
        Constant momentum decay factor
    momentum_history: list
                      Sequence of gradient momentum contributions encountered
                      in the gradient descent
    beta: float
          Constant gradient square sum decay factor
    decay_grad_square_sum_history: list
                         Sequence of decaying sum of squares of gradient
                         encountered in the gradient descent
    """
    super().__init__(f, grad_f, eta)
    self.momentum_history = [0]
    self.decay_grad_square_sum_history = [0]
    self.mu = mu
    self.beta = beta

  def get_mdf(self):
    """
    Returns momentum decay factor for the current iteration. In this version the
    mdf is the constant mu.

    Returns
    -------
    out: float
         Momentum decay factor for the current iteration
    """
    return self.mu

  def get_gssdf(self):
    """
    Returns gradient square sum decay factor for the current iteration.
    In this version the gssdf is the constant beta.

    Returns
    -------
    out: float
         Gradient square sum decay factor for the current iteration
    """
    return self.beta

  def get_lr(self):
    """
    Returns learning rate for the current iteration i. In the adagrad version
    the  learning rate is adaptive for each feature w[k].
    \eta / [sqrt(\sum{1}{i} \beta**i(grad_f(w)[k])**2) / (1-beta)**i]

    Returns
    -------
    out: float
         Learning rate for the current iteration
    """
    w_old = self.history[-1]
    decay_sum_old = self.decay_grad_square_sum_history[-1]
    grad = self.grad_f(w_old)
    decay_sum_new = ((self.get_gssdf() * decay_sum_old)
                     + ((1 - self.get_gssdf()) * grad**2))
    self.decay_grad_square_sum_history.append(decay_sum_new)
    corrected_decay = decay_sum_new / (1 - self.get_gssdf()**len(self.history))
    return self.eta / ((corrected_decay)**0.5 + 1e-10)

  def get_update(self):
    """
    Returns update vector for the current iteration.
    [\mu * momentum_old + (1-\mu) * grad_f(w_old)] / (1-\mu)**{i}
    """
    w_old = self.history[-1]
    momentum_old = self.momentum_history[-1]
    grad = self.grad_f(w_old)
    momentum_new = ((self.get_mdf() * momentum_old)
                  + ((1 - self.get_mdf()) * grad))
    self.momentum_history.append(momentum_new)
    corrected_update = (momentum_new
                        / (1 - self.get_mdf()**len(self.history)))
    update = (self.get_lr() * corrected_update)
    return -update
