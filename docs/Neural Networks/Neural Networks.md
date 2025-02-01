
### History and use cases
The origins:
- How we make software that mimic the brain?
- Started in the 1950's.
- Around 2005 it resurfaced under the name of deep learning.

Use cases:
- Speech
- Images
- Text(NLP)



### Neurons in the brain
![[Pasted image 20240920134129.png]]

- ### Dendrites
	  It is the place where the neuron receives the information.
- ### Axion
	 It is the output of the neuron, here is where the information is transmitted to to other neurons in the form of electrical pulses, these pulses can serve as the input to another neuron.
	

### Artificial networks uses Simplified mathematical model of a neuron.


![[Pasted image 20240920135053.png]]


### Demand Prediction


![[Pasted image 20240920135826.png]]

We can see a logistic regression as a one layer NN.

The sigmoid function is an activation function used in NN to introduce non-linearity to the neuron and learn from complex data.


### Elements of a Neural network.
![[Pasted image 20240920141521.png]]
from : https://galaxyinferno.com/explaining-the-components-of-a-neural-network-ai/

In practice each neuron has access to every feature.

### Neural Network Layer

Multiple layer neural networks has names for each layer
- Layer 0 or Input
- Layer 1 for the to n-1 : hidden layer
- Layer n: output layer.
![[Pasted image 20240920151013.png]]![[Pasted image 20240920151333.png]]
We use superscript notation to index between layers.




### Forward propagation

Forward propagation is the process in a neural network where input data moves through the network's layers to produce an output.


### Model Training Steps

1) The first step is to specify how to compute output given input x and parameters w,b.
2) Specify loss and cost function, the loss function measures the model performance.
3) Minimize the cost function: Use an optimization algorithm like gradient descent.
4) 