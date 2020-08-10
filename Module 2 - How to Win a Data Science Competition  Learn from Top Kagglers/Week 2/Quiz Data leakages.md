## Data leakages
> 
> Latest Submission Grade
> 
> 100%
> 
>  1.Question 1
> 
> Suppose that you have a credit scoring task, where you have to create a ML model that approximates expert evaluation of an individual's creditworthiness. Which of the following can potentially be a data leakage? Select all that apply.
> 
> 2 / 2 points 
> 

      An ID of a data point (row) in the train set correlates with target variable. 
> 
> Check
> 
> Correct
> 
> Data was not shuffled, this information can not be used in real-world scenario
> 

      First half of the data points in the train set has a score of 0, while the second half has scores > 0. 
> 
> Check
> 
> Correct
> 
> Data was not shuffled, this information can not be used in real-world scenario
> 
>  Among the features you have a company_id, an identifier of a company where this person works. It turns out that this feature is very important and adding it to the model significantly improves your score. 
> 
>  2.Question 2
> 
> What is the most foolproof way to set up a time series competition?
> 
> 1 / 1 point 
> 
>  Split train, public and private parts of data by time. Remove time variable from test set, keep the features. 
> 
>  Make a time based split for train/test and a random split for public/private. 
> 

      Split train, public and private parts of data by time. Remove all features except IDs (e.g. timestamp) from test set so that participants will generate all the features based on past and join them themselves. 
> 
> Check
> 
> Correct
> 
> Correct!
> 
>  3.Question 3
> 
> Suppose that you have a binary classification task being evaluated by logloss metric. You know that there are 10000 rows in public chunk of test set and that constant 0.3 prediction gives the public score of 1.01\. Mean of target variable in train is 0.44\. What is the mean of target variable in public part of test data (up to 4 decimal places)?
> 
> 2 / 2 points 
> 

     0.7711
> 
> Check
> 
> Correct
> 
>  4.Question 4
> 
> Suppose that you are solving image classification task. What is the label of this picture?
> 
> ![](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/Po7WII6cEeekTw4xHnjhxA_1349a164252c7f654e705fd2094c88bc_label_is_3.jpg?expiry=1597190400000&hmac=ig60IGaN1efG6WbN_bE3fCn3CUXGq4fg5qafw5DtrSs)
> 
> 1 / 1 point 
> 

     3
> 
> Check
> 
> Correct
>
> -- https://www.coursera.org/learn/competitive-data-science/exam/sPaV5/data-leakages/attempt?redirectToCover=true#Tunnel Vision Close
