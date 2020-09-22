### Graded Advanced Features II Quiz
> 
> Total points 6
> 
> 1.
> 
> Question 1
> 
> Imagine that we apply X = PCA(n_components=5).fit_transform(data) and data has shape (5000, 53). What is the shape of X?
> 
> 1 point
> 
>  (5, 53) 
> 

      (5000, 5) 
> 
>  (5, 5000) 
> 
>  (53, 5) 
> 
> 2.
> 
> Question 2
> 
> To which data NMF is NOT applicable?
> 
> 1 point
> 
>  Bag-of-words matrix 
> 

      Standartized matrix 
> 
>  One-Hot encoded feature 
> 
> 3.
> 
> Question 3
> 
> Suppose we have 2 categorical features: **f1** with A possible values and **f2** with B possible values. How many values will their interaction have?
> 
> 1 point
> 
>  Exactly A + B 
> 
>  Exactly A * B 
> 

      Less or equal to A * B 
> 
>  max(A, B) 
> 
> 4.
> 
> Question 4
> 
> Imagine we have 2 categorical features represented as integers: **f1** with all values in range [0, 1000] and **f2** with values in range [0, 100]. What is the correct way to build their interaction?
> 
> 1 point
> 
>  f1 + f2 
> 
>  f1.astype(str) + f2.astype(str) 
> 

      f1.astype(str) + "_" + f2.astype(str) 
> 
>  (f1 + f2).astype(str) 
> 
> 5.
> 
> Question 5
> 
> What is a correct way to get t-SNE projection of train and test data?
> 
> 1 point
> 
>  Apply t-SNE to the train and after that to the test. 
> 
>  Apply t-SNE to the test first and after to train. 
> 

      Apply t-SNE to concatenation of train and test and split projection back. 
> 
>  Doesn't matter, all variants will produce the same result. 
> 
> 6.
> 
> Question 6
> 
> Is it possible to do t-SNE projection into 20-dimensional space?
> 
> 1 point
> 

      Yes, why not. 
> 
>  No, only 2-dim or 3-dim projections are possible.
>
> -- https://www.coursera.org/learn/competitive-data-science/exam/AGgET/graded-advanced-features-ii-quiz/attempt#Tunnel Vision Close
