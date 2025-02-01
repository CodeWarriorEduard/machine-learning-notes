

We can implement the following neural network  in Tensorflow
like this:
![[Pasted image 20250121215232.png]]


### Input layer
The input layer receives data, we can represent this data using numpy :
```
X = np.array([[200.0, 17.0]])
```

### First hidden layer
This layer is defined using the Dense function in tensorflow.

```
layer_1 = Dense(units=3, activation='sigmoid')
```

### Compute a[1]
```
a1 = layer_1(X)
```

### Second hidden layer

This layer is defined using the Dense function too.


```
layer_2 = Dense(units=1, activation='sigmoid')
a2 = layer_2(a1) // Compute a[2]
```


