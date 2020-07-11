# Modern RNNs
> 
> Latest Submission Grade
> 
> 100%
> 
>  1.Question 1
> 
> Choose correct statements about the exploding gradient problem:
> 
> 1 / 1 point 
> 

      Exploding gradient problem is easy to detect. 
> 
> Check
> 
> Correct
> 
> Exploding gradients are easy to detect, not vanishing.
> 
>  ReLU nonlinearity helps with the exploding gradient problem. 
> 

      The reason of the exploding gradient problem in the simple RNN is the recurrent weight matrix WWW. Nonlinearities sigmoid, tanh, and ReLU does not cause the problem. 
> 
> Check
> 
> Correct
> 
> Derivatives of all these nonlinearities are less than 1, therefore they may cause only the vanishing gradient problem.
> 
>  The threshold for gradient clipping should be as low as possible to make the training more efficient. 
> 
>  2.Question 2
> 
> Choose correct statements about the vanishing gradient problem:
> 
> 1 / 1 point 
> 
>  Vanishing gradient problem is easy to detect. 
> 

      Both nonlinearity and the recurrent weight matrix WWW cause the vanishing gradient problem. 
> 
> Check
> 
> Correct
> 
> That is true!
> 

      Orthogonal initialization of the recurrent weight matrix helps with the vanishing gradient problem. 
> 
> Check
> 
> Correct
> 
> That is true!
> 
>  Truncated BPTT helps with the vanishing gradient problem. 
> 
>  3.Question 3
> 
> Consider the LSTM architecture:
> 
> (gtitotft)=(f~σσσ)(Vxt+Wht−1+b)\left(
> 
> gtitotft
> 
> \begin{array}{c} g_t \\ i_t \\ o_t \\ f_t \end{array} \right) =\left(
> 
> f~σσσ
> 
> \begin{array}{c} \tilde f \\ \sigma \\ \sigma \\ \sigma \end{array} \right) (Vx_t + Wh_{t-1} + b)⎝⎜⎜⎜⎛​gt​it​ot​ft​​⎠⎟⎟⎟⎞​=⎝⎜⎜⎜⎛​f~​σσσ​⎠⎟⎟⎟⎞​(Vxt​+Wht−1​+b)
> 
> ct=ft⋅ct−1+it⋅gt,ht=ot⋅f~(ct)c_t = f_t\cdot c_{t-1} +i_t\cdot g_t,\quad h_t = o_t\cdot \tilde f (c_{t})ct​=ft​⋅ct−1​+it​⋅gt​,ht​=ot​⋅f~​(ct​)
> 
> ![](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/rgAJk58fEee49A5MAYfNMg_12d6c5d789776163bd9b7493fbd5b02f_Screen-Shot-2017-09-22-at-01.49.46.png?expiry=1594598400000&hmac=ZDVVbLoTxPuYnyt7wqttq8TW2NR4j8Y9BU9ijJU_ZUc)
> 
> Choose correct statements about this architecture:
> 
> 1 / 1 point 
> 

      The LSTM needs four times more parameters than the simple RNN. 
> 
> Check
> 
> Correct
> 
> For each gate we need its own set of parameters and there are 3 gates in the LSTM architecture.
> 
>  Gradients do not vanish on the way through memory cells ccc in the LSTM with forget gate. 
> 
>  There is a combination of the gates values which makes the LSTM completely equivalent to the simple RNN. 
> 

      The exploding gradient problem is still possible in LSTM on the way between ht−1h_{t-1}ht−1​ and hth_tht​. 
> 
> Check
> 
> Correct
> 
> Very large norm of WWW may cause the exploding gradient problem. Therefore gradient clipping is useful for LSTM and GRU architectures too.
> 
>  4.Question 4
> 
> Consider the GRU architecture:
> 
> gt=f~(Vgxt+Wg(ht−1⋅rt)+bg)g_t = \tilde f\big(V_gx_t+W_g(h_{t-1}\cdot r_t) + b_g\big)gt​=f~​(Vg​xt​+Wg​(ht−1​⋅rt​)+bg​)
> 
> ht=(1−ut)⋅gt+ut⋅ht−1h_t = (1-u_t)\cdot g_t+u_t\cdot h_{t-1}ht​=(1−ut​)⋅gt​+ut​⋅ht−1​
> 
> ![](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/71G3Bp8cEee49A5MAYfNMg_0a7ca641867c6bdefc3084439990dce4_Screen-Shot-2017-09-22-at-01.30.05.png?expiry=1594598400000&hmac=ubG1vAVGs4iKcVKyRMM2ImdBJSy6KcyB5bJIqszifXA)
> 
> Which combination of the gate values makes this model equivalent to the simple RNN? Here value zero corresponds to a closed gate and value one corresponds to an open gate.
> 
> 1 / 1 point 
> 
>  Both reset and update gates are open. 
> 
>  Both reset and update gates are closed. 
> 

      Reset gate is open and update gate is closed. 
> 
>  Update gate is open and reset gate is closed. 
> 
> Check
> 
> Correct
> 
> That is it!
>
> -- https://www.coursera.org/learn/intro-to-deep-learning/exam/vgWwd/modern-rnns/attempt?redirectToCover=true#Tunnel Vision Close> ## Modern RNNs
>
