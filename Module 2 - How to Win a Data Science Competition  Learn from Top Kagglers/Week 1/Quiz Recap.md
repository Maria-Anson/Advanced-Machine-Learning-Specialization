# Recap
> 
> Total points 4
> 
>  1.Question 1
> 
> Support Vector Machines (SVM) classifier belongs to a class of
> 
> 1 / 1 point 
> 
>  Neural Networks 
> 
>  Nearest Neighbours based 
> 
>  Tree-based models 
> 

      Linear models 
> 
> Check
> 
> Correct
> 
> SVM is a linear model with special loss function. Even with "kernel trick", it's still linear in new, extended space.
> 
>  2.Question 2
> 
> What is the difference between RandomForest and ExtraTrees models from sklearn?
> 
> 1 / 1 point 
> 

      ExtraTrees classifier always tests random splits over fraction of features (in contrast to RandomForest, which tests all possible splits over fraction of features) 
> 
>  ExtraTrees classifier always uses only a fraction of objects when looking for a split (in contrast to Random Forest, which uses all object) 
> 
>  ExtraTrees classifier always uses only a fraction of features when looking for a split (in contrast to Random Forest, which uses all features) 
> 
> Check
> 
> Correct
> 
> Right, this is why they are called extra (randomized) trees
> 
>  3.Question 3
> 
> What model was most probably used to produce such decision surface? Color (from white to purple) shows predicted probability for a point to be of class "red".
> 
> ![](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/R2mkyHqZEeeA3RJRlG3Uqg_4b7f50691fbd29a9de6749e8d9b85f2f_kNN.png?expiry=1596585600000&hmac=ecJTCaw92zTgrdNXM-lbV34aM9l5LJKkEoxp_RG9cbY)
> 
> 1 / 1 point 
> 

      kNN 
> 
>  Linear model 
> 
>  Random Forest 
> 
>  Decision Tree 
> 
> Check
> 
> Correct
> 
> Right. Decision surface is non-linear and does not consist of vertical and horizontal lines, so k-NN is the most plausible option in this list
> 
>  4.Question 4
> 
> What model was most probably used to produce such decision surface? Color (from white to purple) shows predicted probability for a point to be of class "red".
> 
> ![](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/VzLHu3qZEeeJIwrF5BVsIg_503dd71bd77d8526239ec6c194fed45a_logreg.png?expiry=1596585600000&hmac=lFVL8ldeGW2zFFtNOeO6n1BjMMcQz3ofIAuVx5jQK8M)
> 
> 1 / 1 point 
> 
>  k-NN 
> 
>  Random Forest 
> 

      Linear model 
> 
>  Decision Tree 
> 
> Check
> 
> Correct
> 
> Right. Decision boundary is hyperplane, so it was most probably produced by a linear model.
>
> -- https://www.coursera.org/learn/competitive-data-science/quiz/Hwg9B/recap/view-attempt#Tunnel Vision Close
