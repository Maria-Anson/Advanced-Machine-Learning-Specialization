# RNN and Backpropagation
> 
> Latest Submission Grade
> 
> 100%
> 
>  1.Question 1
> 
> Consider a task with input sequences of fixed length. Could RNN architecture still be useful for such task?
> 
> 1 / 1 point 
> 

      Yes 
> 
>  No 
> 
> Check
> 
> Correct
> 
> RNN could be useful since it may need much less number of parameters.
> 
>  2.Question 2
> 
> Consider an RNN for a language generation task. y^t\hat{y}_ty^​t​ is an output of this RNN at each time step, LLL is a length of the input sequence, NNN is a number of words in the vocabulary. Choose correct statements about y^t\hat{y}_ty^​t​:
> 
> 1 / 1 point 
> 

      y^t is a vector of length NNN. 
> 
> Check
> 
> Correct
> 
> The output at each time step is a distribution over a vocabulary, therefore the length of y^t\hat{y}_ty^​t​ is equal to the vocabulary size.
> 
>  y^t is a vector of length (L−t). 
> 
>  y^t is a vector of length L×N. 
> 
>  Each element of y^t is either 0 or 1. 
> 

      Each element of y^t is a number from 0 to 1. 
> 
> Check
> 
> Correct
> 
> Elements of y^t are probabilities so they are numbers from 0 to 1 and the sum of them equal to 1.
> 
>  Each element of y^t is a number from 0 to NNN. 
> 
>  3.Question 3
> 
> Consider the RNN from the lecture:
> 
> ht=ft(Vxt+Wht−1+bh)h_t = f_t(Vx_t+Wh_{t-1}+b_h)ht​=ft​(Vxt​+Wht−1​+bh​)
> 
> y^t=fy(Uht+by)\hat{y}_t = f_y(Uh_t+b_y)y^​t​=fy​(Uht​+by​)
> 
> ![](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/77YTlZvmEeeHrwpWBTEPxg_cd2d0ddc90e794ffa7548a3a2221d5d4_Screen-Shot-2017-09-17-at-23.24.34.png?expiry=1594512000000&hmac=kbbndzYTLI3UzOQoF8bnHKOyNcdk0IZ29mr0MSXZ2fI)
> 
> Calculate the gradient of the loss LLL with respect to the bias vector byb_yby​. ∂L∂by=?\frac{\partial L}{\partial b_y} = \bf{?}∂by​∂L​=?
> 
> 1 / 1 point 
> 

      ∂L∂by=∂L∂y^t∂y^t∂by
> 
>  ∂L∂by=∑t=0T[∂Lt∂y^t∂y^t∂by]\frac{\partial L}{\partial b_y} = \sum_{t=0}^T{\left[\frac{\partial L_t}{\partial \hat{y}_t}\frac{\partial \hat{y}_t}{\partial b_y}\right]} ∂by​∂L​=∑t=0T​[∂y^​t​∂Lt​​∂by​∂y^​t​​] 
> 
>  ∂L∂by=∑t=0T[∂Lt∂y^t∂y^t∂ht∂ht∂by]\frac{\partial L}{\partial b_y} = \sum_{t=0}^T{\left[\frac{\partial L_t}{\partial \hat{y}_t}\frac{\partial \hat{y}_t}{\partial h_t}\frac{\partial h_t}{\partial b_y}\right]} ∂by​∂L​=∑t=0T​[∂y^​t​∂Lt​​∂ht​∂y^​t​​∂by​∂ht​​] 
> 
>  ∂L∂by=∑t=0T[∂Lt∂y^t∂y^t∂ht∑k=0t∂hk∂by]\frac{\partial L}{\partial b_y} = \sum_{t=0}^T{\left[\frac{\partial L_t}{\partial \hat{y}_t}\frac{\partial \hat{y}_t}{\partial h_t}\sum_{k=0}^t{\frac{\partial h_k}{\partial b_y}}\right]} ∂by​∂L​=∑t=0T​[∂y^​t​∂Lt​​∂ht​∂y^​t​​∑k=0t​∂by​∂hk​​] 
> 
>  ∂L∂by=∑t=0T[∂Lt∂y^t∂y^t∂ht∑k=0t(∏i=k+1t∂hi∂hi−1)∂hk∂by]\frac{\partial L}{\partial b_y} = \sum_{t=0}^T{\left[\frac{\partial L_t}{\partial \hat{y}_t}\frac{\partial \hat{y}_t}{\partial h_t}\sum_{k=0}^t{\left(\prod_{i=k+1}^t{\frac{\partial h_i}{\partial h_{i-1}}}\right)\frac{\partial h_k}{\partial b_y}}\right]} ∂by​∂L​=∑t=0T​[∂y^​t​∂Lt​​∂ht​∂y^​t​​∑k=0t​(∏i=k+1t​∂hi−1​∂hi​​)∂by​∂hk​​] 
> 
> Check
> 
> Correct
> 
> It is correct since byb_yby​ influence each LtL_tLt​ only once trough y^t\hat{y}_ty^​t​.
> 
>  4.Question 4
> 
> Consider the RNN network from the previous question. Calculate the gradient of the loss LLL with respect to the bias vector bhb_hbh​. ∂L∂bh=?\frac{\partial L}{\partial b_h} = \bf{?}∂bh​∂L​=?
> 
> 1 / 1 point 
> 
>  ∂L∂bh=∂Lt∂y^t∂y^t∂ht∂ht∂bh\frac{\partial L}{\partial b_h} = \frac{\partial L_t}{\partial \hat{y}_t}\frac{\partial \hat{y}_t}{\partial h_t}\frac{\partial h_t}{\partial b_h}∂bh​∂L​=∂y^​t​∂Lt​​∂ht​∂y^​t​​∂bh​∂ht​​ 
> 
>  ∂L∂bh=∑t=0T[∂Lt∂y^t∂y^t∂ht∂ht∂bh]\frac{\partial L}{\partial b_h} = \sum_{t=0}^T{\left[\frac{\partial L_t}{\partial \hat{y}_t}\frac{\partial \hat{y}_t}{\partial h_t}\frac{\partial h_t}{\partial b_h} \right]}∂bh​∂L​=∑t=0T​[∂y^​t​∂Lt​​∂ht​∂y^​t​​∂bh​∂ht​​] 
> 
>  ∂L∂bh=∑t=0T[∂Lt∂y^t∂y^t∂ht∑k=0t∂hk∂bh]\frac{\partial L}{\partial b_h} = \sum_{t=0}^T{\left[\frac{\partial L_t}{\partial \hat{y}_t}\frac{\partial \hat{y}_t}{\partial h_t}\sum_{k=0}^t{\frac{\partial h_k}{\partial b_h}}\right]} ∂bh​∂L​=∑t=0T​[∂y^​t​∂Lt​​∂ht​∂y^​t​​∑k=0t​∂bh​∂hk​​] 
> 

      ∂L∂bh=∑t=0T[∂Lt∂y^t∂y^t∂ht∑k=0t(∏i=k+1t∂hi∂hi−1)∂hk∂bh]
> 
> Check
> 
> Correct
> 
> It is correct. Hidden units depend on bhb_hbh​ at each time step, therefore we need to backpropagate through time here.
>
> -- https://www.coursera.org/learn/intro-to-deep-learning/exam/NFUgZ/rnn-and-backpropagation/attempt?redirectToCover=true#Tunnel Vision Close
