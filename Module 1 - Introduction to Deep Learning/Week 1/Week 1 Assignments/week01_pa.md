
# Programming assignment (Linear models, Optimization)

In this programming assignment you will implement a linear classifier and train it using stochastic gradient descent modifications and numpy.


```python
import numpy as np
%matplotlib inline
import matplotlib.pyplot as plt
```


```python
import sys
sys.path.append("..")
import grading
grader = grading.Grader(assignment_key="UaHtvpEFEee0XQ6wjK-hZg", 
                      all_parts=["xU7U4", "HyTF6", "uNidL", "ToK7N", "GBdgZ", "dLdHG"])
```


```python
# token expires every 30 min
COURSERA_TOKEN = 'psSLF7X03KieXoke'
COURSERA_EMAIL = 'mariaansona@gmail.com'
```

## Two-dimensional classification

To make things more intuitive, let's solve a 2D classification problem with synthetic data.


```python
with open('train.npy', 'rb') as fin:
    X = np.load(fin)
    
with open('target.npy', 'rb') as fin:
    y = np.load(fin)

print("The number of examples \n",X.shape[0])    
print('Features x1 and x2 for first 20 example \n',X[0:20])
print('Output for first 20 example \n',y[0:20])
```

    The number of examples 
     826
    Features x1 and x2 for first 20 example 
     [[ 1.20798057  0.0844994 ]
     [ 0.76121787  0.72510869]
     [ 0.55256189  0.51937292]
     [-0.58270758  0.26704815]
     [ 2.10228822  1.63387091]
     [ 0.32010071  0.02504327]
     [ 0.62014803  1.81595317]
     [-0.834067    0.81647546]
     [ 1.16340279  0.65438798]
     [-0.47376216  1.04974385]
     [ 0.60334006  0.13730586]
     [-1.99202745 -0.6744618 ]
     [-0.13720399  0.17291083]
     [ 0.8660117   1.37552587]
     [ 0.286358   -0.36087453]
     [-1.88537687 -0.3179157 ]
     [-2.1673067  -1.29643564]
     [ 1.91280872 -0.49738253]
     [ 1.00217947  0.39072288]
     [-1.29605598 -2.37806727]]
    Output for first 20 example 
     [1 1 1 1 0 1 0 1 0 1 1 0 1 0 1 0 0 1 1 0]



```python
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired, s=10)
plt.xlabel('x1',fontsize=20)
plt.ylabel('x2',fontsize=20)
plt.show()
```


![png](output_6_0.png)


# Task

## Features

As you can notice the data above isn't linearly separable. Since that we should add features (or use non-linear model). Note that decision line between two classes have form of circle, since that we can add quadratic features to make the problem linearly separable. The idea under this displayed on image below:

![](kernel.png)


```python
def expand(X):
    """
    Adds quadratic features. 
    This expansion allows your linear model to make non-linear separation.
    
    For each sample (row in matrix), compute an expanded row:
    [feature0, feature1, feature0^2, feature1^2, feature0*feature1, 1]
    
    :param X: matrix of features, shape [n_samples,2]
    :returns: expanded features of shape [n_samples,6]
    """
    X_expanded = np.zeros((X.shape[0], 6))
    feature0 = X[:,0].reshape(-1,1)
    feature1 = X[:,1].reshape(-1,1)
    feature0_2 = (X[:,0]**2).reshape(-1,1)
    feature1_2 = (X[:,1]**2).reshape(-1,1)
    feature0_feature1 = (X[:,0]*X[:,1]).reshape(-1,1)
    one = np.ones((X.shape[0],1))
    
    X_expanded = np.concatenate([feature0,feature1,feature0_2,feature1_2,feature0_feature1,one], axis = 1) 
    
    return X_expanded
```


```python
X_expanded = expand(X)
X_expanded
```




    array([[ 1.20798057,  0.0844994 ,  1.45921706,  0.00714015,  0.10207364,
             1.        ],
           [ 0.76121787,  0.72510869,  0.57945265,  0.52578261,  0.5519657 ,
             1.        ],
           [ 0.55256189,  0.51937292,  0.30532464,  0.26974823,  0.28698568,
             1.        ],
           ..., 
           [-1.22224754,  0.45743421,  1.49388906,  0.20924606, -0.55909785,
             1.        ],
           [ 0.43973452, -1.47275142,  0.19336645,  2.16899674, -0.64761963,
             1.        ],
           [ 1.4928118 ,  1.15683375,  2.22848708,  1.33826433,  1.72693508,
             1.        ]])




```python
X_expanded.shape
```




    (826, 6)



Here are some tests for your implementation of `expand` function.


```python
# simple test on random numbers

dummy_X = np.array([
        [0,0],
        [1,0],
        [2.61,-1.28],
        [-0.59,2.1]
    ])

# call your expand function
dummy_expanded = expand(dummy_X)

# what it should have returned:   x0       x1       x0^2     x1^2     x0*x1    1
dummy_expanded_ans = np.array([[ 0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  1.    ],
                               [ 1.    ,  0.    ,  1.    ,  0.    ,  0.    ,  1.    ],
                               [ 2.61  , -1.28  ,  6.8121,  1.6384, -3.3408,  1.    ],
                               [-0.59  ,  2.1   ,  0.3481,  4.41  , -1.239 ,  1.    ]])

#tests
assert isinstance(dummy_expanded,np.ndarray), "please make sure you return numpy array"
assert dummy_expanded.shape == dummy_expanded_ans.shape, "please make sure your shape is correct"
assert np.allclose(dummy_expanded,dummy_expanded_ans,1e-3), "Something's out of order with features"

print("Seems legit!")

```

    Seems legit!


## Logistic regression

To classify objects we will obtain probability of object belongs to class '1'. To predict probability we will use output of linear model and logistic function:

$$ a(x; w) = \langle w, x \rangle $$
$$ P( y=1 \; \big| \; x, \, w) = \dfrac{1}{1 + \exp(- \langle w, x \rangle)} = \sigma(\langle w, x \rangle)$$



```python
def probability(X, w):
    """
    Given input features and weights
    return predicted probabilities of y==1 given x, P(y=1|x), see description above
        
    Don't forget to use expand(X) function (where necessary) in this and subsequent functions.
    
    :param X: feature matrix X of shape [n_samples,6] (expanded)
    :param w: weight vector w of shape [6] for each of the expanded features
    :returns: an array of predicted probabilities in [0,1] interval.
    """
    
    return 1/(1+np.exp(-np.dot(X,w)))
```


```python
print("Sample input first row {} \nIt's shape {}".format(X_expanded[:1, :], X_expanded[:1, :].shape))
print("Sample weight {} \nIt's shape {}".format(np.linspace(-1,1,6), np.linspace(-1,1,6).shape))
```

    Sample input first row [[ 1.20798057  0.0844994   1.45921706  0.00714015  0.10207364  1.        ]] 
    It's shape (1, 6)
    Sample weight [-1.  -0.6 -0.2  0.2  0.6  1. ] 
    It's shape (6,)



```python
dummy_weights = np.linspace(-1, 1, 6)
ans_part1 = probability(X_expanded[:1, :], dummy_weights)[0]

```


```python
ans_part1
```




    0.3803998509843769




```python
## GRADED PART, DO NOT CHANGE!
grader.set_answer("xU7U4", ans_part1)
```


```python
# you can make submission with answers so far to check yourself at this stage
grader.submit(COURSERA_EMAIL, COURSERA_TOKEN)
```

    Submitted to Coursera platform. See results on assignment page!


In logistic regression the optimal parameters $w$ are found by cross-entropy minimization:

Loss for one sample: $$ l(x_i, y_i, w) = - \left[ {y_i \cdot log P(y_i = 1 \, | \, x_i,w) + (1-y_i) \cdot log (1-P(y_i = 1\, | \, x_i,w))}\right] $$

Loss for many samples: $$ L(X, \vec{y}, w) =  {1 \over \ell} \sum_{i=1}^\ell l(x_i, y_i, w) $$




```python
def compute_loss(X, y, w):
    """
    Given feature matrix X [n_samples,6], target vector [n_samples] of 1/0,
    and weight vector w [6], compute scalar loss function L using formula above.
    Keep in mind that our loss is averaged over all samples (rows) in X.
    """
    l = X.shape[0]
    hypothesis = probability(X,w)
    loss = np.dot(-y,np.log(hypothesis)) - np.dot((1-y),np.log(1-hypothesis))  
    loss = loss/l
    
    return loss
```


```python
# use output of this cell to fill answer field 
ans_part2 = compute_loss(X_expanded, y, dummy_weights)
```


```python
## GRADED PART, DO NOT CHANGE!
grader.set_answer("HyTF6", ans_part2)
```


```python
# you can make submission with answers so far to check yourself at this stage
grader.submit(COURSERA_EMAIL, COURSERA_TOKEN)
```

    Submitted to Coursera platform. See results on assignment page!


Since we train our model with gradient descent, we should compute gradients.

To be specific, we need a derivative of loss function over each weight [6 of them].

$$ \nabla_w L = {1 \over \ell} \sum_{i=1}^\ell \nabla_w l(x_i, y_i, w) $$ 

We won't be giving you the exact formula this time — instead, try figuring out a derivative with pen and paper. 

As usual, we've made a small test for you, but if you need more, feel free to check your math against finite differences (estimate how $L$ changes if you shift $w$ by $10^{-5}$ or so).


```python
def compute_grad(X, y, w):
    """
    Given feature matrix X [n_samples,6], target vector [n_samples] of 1/0,
    and weight vector w [6], compute vector [6] of derivatives of L over each weights.
    Keep in mind that our loss is averaged over all samples (rows) in X.
    """
    
    l = X.shape[0]
    grad = (1/l)*np.dot((probability(X,w)-y).T,X)
    
    return grad
```


```python
# use output of this cell to fill answer field 
ans_part3 = np.linalg.norm(compute_grad(X_expanded, y, dummy_weights))
```


```python
## GRADED PART, DO NOT CHANGE!
grader.set_answer("uNidL", ans_part3)
```


```python
# you can make submission with answers so far to check yourself at this stage
grader.submit(COURSERA_EMAIL, COURSERA_TOKEN)
```

    Submitted to Coursera platform. See results on assignment page!


Here's an auxiliary function that visualizes the predictions:


```python
from IPython import display

h = 0.01
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

def visualize(X, y, w, history):
    """draws classifier prediction with matplotlib magic"""
    Z = probability(expand(np.c_[xx.ravel(), yy.ravel()]), w)
    Z = Z.reshape(xx.shape)
    plt.subplot(1, 2, 1)
    plt.contourf(xx, yy, Z, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    
    plt.subplot(1, 2, 2)
    plt.plot(history)
    plt.grid()
    ymin, ymax = plt.ylim()
    plt.ylim(0, ymax)
    display.clear_output(wait=True)
    plt.show()
```


```python
visualize(X, y, dummy_weights, [0.5, 0.5, 0.25])
```


![png](output_32_0.png)


## Training
In this section we'll use the functions you wrote to train our classifier using stochastic gradient descent.

You can try change hyperparameters like batch size, learning rate and so on to find the best one, but use our hyperparameters when fill answers.

## Mini-batch SGD

Stochastic gradient descent just takes a random batch of $m$ samples on each iteration, calculates a gradient of the loss on it and makes a step:
$$ w_t = w_{t-1} - \eta \dfrac{1}{m} \sum_{j=1}^m \nabla_w l(x_{i_j}, y_{i_j}, w_t) $$




```python
batch_size = 4
ind = np.random.choice(X_expanded.shape[0], batch_size)
print("Random 4 rows",ind)
```

    Random 4 rows [739 810 147 698]



```python
# please use np.random.seed(42), eta=0.1, n_iter=100 and batch_size=4 for deterministic results

np.random.seed(42)
w = np.array([0, 0, 0, 0, 0, 1])

eta= 0.1 # learning rate

n_iter = 100
batch_size = 4
loss = np.zeros(n_iter)
plt.figure(figsize=(12, 5))

for i in range(n_iter):
    ind = np.random.choice(X_expanded.shape[0], batch_size)
    loss[i] = compute_loss(X_expanded, y, w)
    if i % 10 == 0:
        visualize(X_expanded[ind, :], y[ind], w, loss)

    # Keep in mind that compute_grad already does averaging over batch for you!
    w = w - eta*compute_grad(X_expanded[ind,:],y[ind],w)

visualize(X, y, w, loss)
plt.clf()
```


![png](output_36_0.png)



    <matplotlib.figure.Figure at 0x7f5c3f72f3c8>



```python
# use output of this cell to fill answer field 

ans_part4 = compute_loss(X_expanded, y, w)
```


```python
## GRADED PART, DO NOT CHANGE!
grader.set_answer("ToK7N", ans_part4)
```


```python
# you can make submission with answers so far to check yourself at this stage
grader.submit(COURSERA_EMAIL, COURSERA_TOKEN)
```

    Submitted to Coursera platform. See results on assignment page!


## SGD with momentum

Momentum is a method that helps accelerate SGD in the relevant direction and dampens oscillations as can be seen in image below. It does this by adding a fraction $\alpha$ of the update vector of the past time step to the current update vector.
<br>
<br>

$$ \nu_t = \alpha \nu_{t-1} + \eta\dfrac{1}{m} \sum_{j=1}^m \nabla_w l(x_{i_j}, y_{i_j}, w_t) $$
$$ w_t = w_{t-1} - \nu_t$$

<br>


![](sgd.png)



```python
# please use np.random.seed(42), eta=0.05, alpha=0.9, n_iter=100 and batch_size=4 for deterministic results
np.random.seed(42)
w = np.array([0, 0, 0, 0, 0, 1])

eta = 0.05 # learning rate
alpha = 0.9 # momentum
nu = np.zeros_like(w)

n_iter = 100
batch_size = 4
loss = np.zeros(n_iter)
plt.figure(figsize=(12, 5))

for i in range(n_iter):
    ind = np.random.choice(X_expanded.shape[0], batch_size)
    loss[i] = compute_loss(X_expanded, y, w)
    if i % 10 == 0:
        visualize(X_expanded[ind, :], y[ind], w, loss)

    
    nu = alpha*nu + eta*compute_grad(X_expanded[ind,:],y[ind],w)
    w = w-nu
visualize(X, y, w, loss)
plt.clf()
```


![png](output_41_0.png)



    <matplotlib.figure.Figure at 0x7f5bf8045f28>



```python
# use output of this cell to fill answer field 

ans_part5 = compute_loss(X_expanded, y, w)
```


```python
## GRADED PART, DO NOT CHANGE!
grader.set_answer("GBdgZ", ans_part5)
```


```python
# you can make submission with answers so far to check yourself at this stage
grader.submit(COURSERA_EMAIL, COURSERA_TOKEN)
```

    Submitted to Coursera platform. See results on assignment page!


## RMSprop

Implement RMSPROP algorithm, which use squared gradients to adjust learning rate:

$$ G_j^t = \alpha G_j^{t-1} + (1 - \alpha) g_{tj}^2 $$
$$ w_j^t = w_j^{t-1} - \dfrac{\eta}{\sqrt{G_j^t + \varepsilon}} g_{tj} $$


```python
# please use np.random.seed(42), eta=0.1, alpha=0.9, n_iter=100 and batch_size=4 for deterministic results
np.random.seed(42)

w = np.array([0, 0, 0, 0, 0, 1.])

eta = 0.1 # learning rate
alpha = 0.9 # moving average of gradient norm squared
g_prev = None # we start with None so that you can update this value correctly on the first iteration
eps = 1e-8

n_iter = 100
batch_size = 4
loss = np.zeros(n_iter)
plt.figure(figsize=(12,5))

g_current = compute_grad(X_expanded, y, w)
g_prev = g_current*g_current
g_prev = alpha * g_prev + (1 - alpha) * np.square(g_current)
w = w - eta * g_current / np.sqrt(g_prev + eps)

for i in range(n_iter):
    ind = np.random.choice(X_expanded.shape[0], batch_size)
    loss[i] = compute_loss(X_expanded, y, w)
    if i % 10 == 0:
        visualize(X_expanded[ind, :], y[ind], w, loss)

    g_current = compute_grad(X_expanded[ind,:],y[ind],w)
    g_prev = alpha*g_prev + (1-alpha)*(np.square(g_current))
    w =  w - eta*g_current /np.sqrt(g_prev+eps)

visualize(X, y, w, loss)
plt.clf()
```


![png](output_46_0.png)



    <matplotlib.figure.Figure at 0x7f5befbba7f0>



```python
# use output of this cell to fill answer field 
ans_part6 = compute_loss(X_expanded, y, w)
```


```python
## GRADED PART, DO NOT CHANGE!
grader.set_answer("dLdHG", ans_part6)
```


```python
grader.submit(COURSERA_EMAIL, COURSERA_TOKEN)
```

    Submitted to Coursera platform. See results on assignment page!

