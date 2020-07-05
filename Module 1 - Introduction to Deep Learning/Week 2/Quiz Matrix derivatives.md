# Matrix derivatives
> 
> Total points 3
> 
>  1.Question 1
> 
> Choose the correct statements about MLP implementation:
> 
> 1 point 
> 
>  You shouldn't prefer matrix operations when working with GPU 
> 

      A forward pass of a dense layer can be done with matrix product 
> 

      You can write both passes of a dense layer with NumPy and make it quick even in Python 
> 
>  A backward pass of a dense layer needs a 4-d tensor derivative 
> 
>  2.Question 2
> 
> How many dimensions will a derivative of a 3-d tensor by a 4-d tensor have?
> 

      7
> 1 point 
> 
>  3.Question 3
> 
> Let's play around with matrix derivatives!
> 
> A trace Tr(X)Tr(X)Tr(X) of a matrix XXX is a sum of its diagonal elements.
> 
> For example: Tr(1331)=1+1=2Tr\left(
> 
> 1331
> 
> \begin{matrix}1 & 3 \\ 3 & 1 \end{matrix}\right) = 1 + 1 = 2Tr(13​31​)=1+1=2. Note that trace is a scalar!
> 
> Let's find a matrix notation for ∂Tr(X2)∂X\frac{\partial Tr(X^2)}{\partial X}∂X∂Tr(X2)​ for matrix X=(x1,1x1,2x2,1x2,2) X = \left(
> 
> x1,1x2,1x1,2x2,2
> 
> \begin{matrix}x_{1,1} & x_{1,2} \\ x_{2,1} & x_{2,2} \end{matrix}\right)X=(x1,1​x2,1​​x1,2​x2,2​​), where X2X^2X2 is a matrix product X⋅X X \cdot XX⋅X.
> 
> Please do this element-wise and figure out a matrix notation for it:
> 
> 1 point 
> 
>  2Tr(XT) 2Tr(X^T) 2Tr(XT) 
> 
>  Tr(2X) Tr(2X) Tr(2X) 
> 
>  XTX X^TX XTX 
> 

      2X^T
> 
>  2X 2X 2X
>
> -- https://www.coursera.org/learn/intro-to-deep-learning/exam/0kXXi/matrix-derivatives/attempt#Tunnel Vision Close
