### Exploratory data analysis
> 
> Latest Submission Grade
> 
> 100%
> 
>  1.Question 1
> 
> ![](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/1tVFknsvEeeOygpRbdVQKg_338ab2d607d26a15cf876118de5de546_q2.png?expiry=1597104000000&hmac=DIb2hplTNIIsas9LpvXF1rUn6eczV9v9tKKhvaM4kxY)
> 
> Suppose we are given a data set with features XXX, YYY, ZZZ.
> 
> On the top figure you see a scatter plot for variables XXX and YYY. Variable ZZZ is a function of XXX and YYY and on the bottom figure a scatter plot between XXX and ZZZ is shown. Can you recover ZZZ as a function of XXX and YYY?
> 
> 2 / 2 points 
> 

      Z=X/Y 
> 
>  Z=X−YZ = X - YZ=X−Y 
> 
>  Z=XYZ = XYZ=XY 
> 
>  Z=X+YZ = X + YZ=X+Y 
> 
> Check
> 
> Correct
> 
> Correct!
> 
>  2.Question 2
> 
> ![](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/HoKBcXsyEee6mw7xN92yoA_11bfb551f18194ec117157caf2aac94e_q3.png?expiry=1597104000000&hmac=t-03N8W96FfuA_gl8_ltpiX-eKNr-r4KRbAr5VZXysI)
> 
> What YYY value do the objects colored in red have?
> 
> 2 / 2 points 
> 

     2
> 
> Check
> 
> Correct
> 
> The equation for a line, built through red points is Z=X/2Z = X/2Z=X/2, now recalling that Z=X/YZ = X/YZ=X/Y we conclude Y=2Y = 2Y=2.
> 
>  3.Question 3
> 
> ![](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/7e1VE3swEeeJIwrF5BVsIg_f6287023139779eb05887e554451f947_q1.png?expiry=1597104000000&hmac=rXU-bCDKiLH6v4dFwlyyhiLuOr2JrB2p2gdHxA5TqDE)
> 
> The following code was used to produce these two plots:
> 
> <pre contenteditable="false" data-language="python" style="opacity: 1;" tabindex="0">
> 
> 
> # top plot
> 
> plt.plot(x, '.')
> 
> # bottom plot
> 
> logX = np.log1p(x) # no NaNs after this operation
> 
> plt.plot(logX, '.')
> 
> 
> (note that it is not the same variable XXX as in previous questions).
> 
> Which hypotheses about variable XXX do NOT contradict with the plots? In other words: what hypotheses we can't reject (not in statistical sense) based on the plots and our intuition?
> 
> 2 / 2 points 
> 
>  XXX can be the temperature (in Celsius) in different cities at different times 
> 

      2≤X<32\leq X<32≤X<3 happens more frequently than 3≤X<43\leq X<43≤X<4 
> 
> Check
> 
> Correct
> 
> Yes! It can be the case, we cannot understand it from these plots, more exploration is needed, but such hypothesis does not contradicts with the plots.
> 
>  XXX can take a value of zero 
> 

      XXX takes only discrete values 
> 
> Check
> 
> Correct
> 
> In fact, horizontal lines indicate a lot or repeated values. The most bottom horizontal line on log(X+1)log(X+1)log(X+1) plot corresponds to the value 111, the next to the value 222 and so on.
> 

      XXX is a counter or label encoded categorical feature 
> 
> Check
> 
> Correct
> 
> Yes! The values are integers and start from 111. It could be e.g. a counter how many times a used opened web-site. Or it could be a a categorical features encoded with label encoder, which starts with label 111 (in pandas and sklearn label encoders usually start with 000).
> 
>  4.Question 4
> 
> ![](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/U7JlBHs2EeeA3RJRlG3Uqg_4c17f0f64005cb0cdbc7957d93598bb5_q4.png?expiry=1597104000000&hmac=GldrJrJlt2spkNMgk9amjSE2f9ORGYz-PewAD0lbvno)
> 
> Suppose we are given a dataset with features XXX and YYY and need to learn to classify objects into 222 classes. The corresponding targets for the objects from the dataset are denoted as yyy.
> 
> Top left plot shows XXX vs YYY scatter plot, produced with the following code:
> 
> 
> # y is a target vector
> 
> plt.scatter(X, Y, c = y)
> 
> 
> We use target variable yyy to colorcode the points.
> 
> The other three plots were produced by _jittering_ XXX and YYY values:
> 
> 
> def jitter(data, stdev):
> 
>   N = len(data)
> 
>   return data + np.random.randn(N) * stdev
> 
> # sigma is a given std. dev. for Gaussian distribution
> 
> plt.scatter(jitter(X, sigma), jitter(Y, sigma), c = y)
> 
> 
> That is, we add Gaussian noise to the features before drawing scatter plot.
> 
> Select the correct statements.
> 
> 2 / 2 points 
> 

      Standard deviation for Jittering is the largest on the bottom right plot. 
> 
> Check
> 
> Correct
> 
> Yes! We can't even see, that XXX, YYY originally have small number of unique values.
> 
>  We need to jitter variables not only for a sake of visualization, but also because it is beneficial for a model. 
> 
>  It is _always_ beneficial to jitter variables before building a scatter plot 
> 

      Top right plot is "better" than top left one. That is, every piece of information we can find on the top left we can also find on the top right, but not vice versa. 
> 
> Check
> 
> Correct
> 
> Yes! On the top left plot we only see, that pairs (x,y)(x,y)(x,y) lie on the grid. Top right also shows target distribution for each (x,y)(x,y)(x,y) and density in (x,y)(x,y)(x,y).
> 
>  Target is completely determined by coordinates (x,y)(x,y)(x,y), i.e. the label of the point is _completely determined_ by point's position (x,y)(x,y)(x,y). Saying the same in other words: if we only had two features (x,y)(x,y)(x,y), we could build a classifier, that is accurate 100% of time.
>
> -- https://www.coursera.org/learn/competitive-data-science/exam/AjWOf/exploratory-data-analysis/attempt?redirectToCover=true#Tunnel Vision Close
