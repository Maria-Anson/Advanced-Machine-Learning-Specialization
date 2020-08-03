# Recap
> 
> Total points 6
> 
>  1.Question 1
> 
> What back propagation is usually used for in neural networks?
> 
> 1 point 
> 
>  Select gradient update direction by flipping a coin 
> 
>  Make several random perturbations of parameters and go back to the best one 
> 

      To calculate gradient of the loss function with respect to the parameters of the network 
> 
>  To propagate signal through network from input to output only 
> 
>  2.Question 2
> 
> Suppose we've trained a RandomForest model with 100 trees. Consider two cases:
> 
> 1.  We drop the first tree in the model
> 2.  We drop the last tree in the model
> 
> We then compare models performance _on the train set_. Select the right answer.
> 
> 1 point 
> 

      In the _case 1_ performance **will be roughly the same** as in the _case 2_ 
> 
>  In the _case 1_ performance **will drop more** than in the _case 2_ 
> 
>  In the _case 1_ performance **will drop less** than in the _case 2_ 
> 
>  3.Question 3
> 
> Suppose we've trained a GBDT model with 100 trees with a fairly large learning rate. Consider two cases:
> 
> 1.  We drop the first tree in the model
> 2.  We drop the last tree in the model
> 
> We then compare models performance _on the train set_. Select the right answer.
> 
> 1 point 
> 

     In the _case 1_ performance **will drop more** than in the _c__ase 2_ 
> 
>  In the _case 1_ performance **will be roughly the same** as in the _case 2_ 
> 
>  In the _case 1_ performance **will drop less** than in the _c__ase 2_ 
> 
>  4.Question 4
> 
> Consider two cases:
> 
> 1.  We fit two RandomForestClassifiers 500 trees each and average their predicted probabilities on the test set.
> 2.  We fit a RandomForestClassifier with 1000 trees and use it to get test set probabilities.
> 
> All hyperparameters except number of trees are the same for all models.
> 
> Select the right answer.
> 
> 1 point 
> 
>  The quality of predictions in the _case 1_ **will be higher** than the quality of the predictions in the _case 2_ 
> 
>  The quality of predictions in the _case 1_ **will be lower** than the quality of the predictions in the _case 2_ 
> 

      The quality of predictions in the _case 1_ **will be roughly the same** as the quality of the predictions in the _case 2_ 
> 
>  5.Question 5
> 
> What model was most probably used to produce such decision surface? Color (from white to purple) shows predicted probability for a point to be of class "red".
> 
> ![](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/_LUm1XqXEeeA3RJRlG3Uqg_56a252e3d83d1655792bfd4da667310e_dt.png?expiry=1596585600000&hmac=HCBa0ycYfssu-11ZzYXHlExSmPGPzQq-TTU0PMyEU30)
> 
> 1 point 
> 
>  Linear model 
> 
>  Random Forest 
> 
>  k-NN 
> 

      Decision Tree 
> 
>  6.Question 6
> 
> What model was most probably used to produce such decision surface?
> 
> ![](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/ntCFVnqZEee6QxIw_sWf5g_1d788ff8639da8a2e6c564ae12e6f13a_rf.png?expiry=1596585600000&hmac=c4i8kEIJAtfEZlMthIGR5hMtt5ZfAB0lUUq4-vl3UNE)
> 
> 1 point 
> 
>  k-NN 
> 
>  Decision Tree 
> 

      Random Forest 
> 
>  Linear model
>
> -- https://www.coursera.org/learn/competitive-data-science/exam/XWR9K/recap/attempt#Tunnel Vision Close
