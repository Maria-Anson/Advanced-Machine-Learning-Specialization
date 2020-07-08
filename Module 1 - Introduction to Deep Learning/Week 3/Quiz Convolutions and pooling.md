# Convolutions and pooling
> 
> Latest Submission Grade
> 
> 100%
> 
>  1.Question 1
> 
> Choose correct statements about convolutional layer:
> 
> 1 / 1 point 
> 
>  Convolutional layer doesn't need a bias term 
> 
>  Convolutional layer provides translation invariance 
> 

      Convolutional layer works the same way for every input patch 
> 
> Check
> 
> Correct
> 
> Because kernel parameters are shared!
> 

      Convolutional layer is a special case of a fully-connected layer 
> 
> Check
> 
> Correct
> 
> Convolutional layer can be viewed as a special case of a fully connected layer when all the weights outside the local receptive field of each output neuron equal 0 and kernel parameters are shared between neurons
> 
>  2.Question 2
> 
> Choose correct statements about pooling layer:
> 
> 1 / 1 point 
> 

      Pooling layer provides translation invariance 
> 
> Check
> 
> Correct
> 
> Remember the slash classifier example? Taking maximum gave us translation invariance.
> 

      Pooling layer can reduce spatial dimensions (width and height of the input volume) 
> 
> Check
> 
> Correct
> 
> When used with stride > 1
> 
>  Pooling layer reduces the number of convolutional filters 
> 
>  Pooling layer is strictly differentiable 
> 
>  3.Question 3
> 
> Back-propagation for convolutional layer first calculates the gradients as if the kernel parameters were not shared and then...
> 
> 1 / 1 point 
> 

      Takes a sum of gradients for each shared parameter 
> 
>  Takes a mean of the gradients for each shared parameter 
> 
>  Takes a maximum gradient for each shared parameter 
> 
>  Takes a minimum gradient for each shared parameter 
> 
> Check
> 
> Correct
> 
> That's it!
> 
>  4.Question 4
> 
> Suppose you have a 10x10x3 colour image input and you want to stack two convolutional layers with kernel size 3x3 with 10 and 20 filters respectively. How many parameters do you have to train for these two layers? Don't forget bias terms!
> 
> 1 / 1 point 
> 
> Preview
> 
> 2100
> 

     (3*3*3+1)*10+(3*3*10+1)*20
> 
> Check
> 
> Correct
> 
> (3*3*3+1)*10 + (3*3*10+1)*20
> 
>  5.Question 5
> 
> What receptive field do we have after stacking nnn convolutional layers with kernel size k×kk \times kk×k and stride 1? Layers numeration starts with 1\. The resulting receptive field will be a square, input its side as an answer.
> 
> 1 / 1 point 
> 
> Preview
> 
> kn−n+1k n - n + 1kn−n+1
> 

     n*k-n+1
> 
> Check
> 
> Correct
>
> -- https://www.coursera.org/learn/intro-to-deep-learning/exam/gy9pN/convolutions-and-pooling/attempt?redirectToCover=true#Tunnel Vision Close
