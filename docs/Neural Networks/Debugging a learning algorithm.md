![[Pasted image 20250129160053.png]]

Lets suppose we implemented a regularized linear regression on housing prices and we observe that it makes large errors in predictions.

We can try these things:
- More training examples
- Try smaller sets of features
- Additional features
- Polynomial features
- Decreasing or increasing lambda.


### Machine learning diagnostic

Diagnostic: A test that you run to gain insight into what is /isn't working with a learning algorithm, to gain guidance into improving its performance.

Diagnostics can take time to implement but doing so can be a very good use of your time.


### Evaluating a model
![[Pasted image 20250129161529.png]]
In this case model fits the training data well, but will fail to generalize to new examples not in the training set.
Why? We are using to predict the price just the size of the size...

We need to include more features like:

- No. of bedrooms
- No. of floors
- Age of home in years


To evaluate our model we can split the training set into two subsets: 
- Test set 
- Training set.

![[Pasted image 20250129165759.png]]


### Model selection
![[Pasted image 20250130103648.png]]

An often technique used to select a model is cross validation. This technique consists in splitting the set in 3 parts, training, cv and test set. We compute the error in the cross validation set and the model with the lowest cross-validation error is the one that we need to choose.
We perform Jtest to estimate the generalization error.

The same occurs when choosing a neural network architecture
![[Pasted image 20250130104427.png]]


