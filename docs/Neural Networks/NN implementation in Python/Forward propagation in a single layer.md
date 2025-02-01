
In this section we are going to implement forward propagation from scratch for a single layer in python.

Forward propagation is the process of passing data through the neural network.



![[Pasted image 20250122102529.png]]

The first value to calculate is a^[1] 

![[Pasted image 20250122102912.png]]

To calculate this value we need to compute  the activation function given by the dot product of the weight by the input plus the bias like this:

Given the following weight and a bias:

```

w1_1 = np.array([1, 2])
b1_1 = np.array([-1])

```

We can now calculate the value of z and the activation function

```
z1_1 = np.dot(w1_1, x) + b1_1
a1_1 = sigmoid(z1_1)
```


And so on with the other neurons.

For neuron 2:
```

w1_2 = np.array([-3, 4])
b1_2 = np.array([1])
z1_2 = np.dot(w1_2, x) + b1_2
a1_2 = sigmoid(z1_2)

```

For neuron 3:
```

w1_3 = np.array([5, -6])
b1_3 = np.array([2])
z1_3 = np.dot(w1_3, x) + b1_2
a1_3 = sigmoid(z1_3)

```


So a^[1] equals to

```
a1 = np.array([a1_1, a1_2, a1_3])
```

On layer 2

To calculate a^[2]  we use the following formula
![[Pasted image 20250122111307.png]]


```
w2_1 = np.array([-7, 8, 9])
b2_1 = np.array([3])
z2_1 = np.dot(w2_1, a1) + b2_1
a2_1 = sigmoid(z2_1)
```


### General implementation

Let's define a function to implement a dense layer a dense layer represents a single layer of a neural network. This dense function takes as input the activation from the previous layer, the w parameters stacked in columns and b parameters stacked into a 1D array.

![[Pasted image 20250122115616.png]]


```
def dense(a_in, W, b):
	units = W.shape[1]
	a_out = np.zeros(units)
	for j in range(units):
	w = W[:, j]
	z = np.dot(w, a_in) + b[j]
	a_out[j] = g(z)
return a_out
```


Now in order to perform forward propagation through all the layers we define a function called sequentially.

```
def sequential(x):
	a1 = dense(x, W1, b1)
	a2 = dense(a1, W2, b2)
	a3 = dense(a2, W3, b3)
    a4 = dense(a3, W4, b4)
    f_x = a4
    return f_x
 
```


### Vectorization

Vectorization allows to optimize operations happening in a neural network by computing multiple sets of data at the same time.
![[Pasted image 20250122122819.png]]
