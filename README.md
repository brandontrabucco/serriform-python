# The Serriform Neural Network

![The Algorithm Expression](https://trello-attachments.s3.amazonaws.com/5741d22c3d23ec58c253c910/5951950b218e8069e6e63b79/cb39f4c32e849ca4cb893a8954f7bb6c/image.png)

The • symbol denotes a dot product.

The : symbol denotes a tensor double-contraction

The ⊙ symbol denotes element-wise multiplication.

This algorithm is an extension of the fully-connected feed-forward artificial neural networks. For this algorithm, neurons in a layer ℓ may connect to any neurons in any layer (ℓ + n) where n is any positive non-zero integer. Additionally, a delay of 1 update cycle is set between neurons. A delay allows for each connection in this algorithm to be updated simultaneously.

Overlapping delayed neuron connections allow this algorithm to learn patterns within a fixed temporal window. This window is given by the amount of overlap between neuron connections. Unlike other sequence learning algorithms, no recurrent neuron connections are used in this algorithm, and a modified version of back-propagation though time allows for online training.

![The Serriform Neural Network](https://trello-attachments.s3.amazonaws.com/5741d22c3d23ec58c253c910/5951950b218e8069e6e63b79/78723ce71b6b4e4f9c36309661192752/serriform-net.png)

This image is a generalized structure for the Serriform Neural Network. Each green dot represents an input neuron. Each red dot represents a state neuron. Each blue dot is an output neuron. Each green line represents an input connector. Each red line is a state connector. Each blue line is an output connector. Each connector has an associated weight.

## The Activation Function

![The Activation Function](https://trello-attachments.s3.amazonaws.com/5741d22c3d23ec58c253c910/5951950b218e8069e6e63b79/352e82069f797491df91812a770e8c74/image.png)

The standard logistic curve is used as the activation function of this algorithm, and is denoted by σ(x) in the above expression. The derivative of this function with respect to the input is expressed as σ(x) * (1 - σ(x)).

## The Input Matrix

![The Input Matrix](https://trello-attachments.s3.amazonaws.com/5741d22c3d23ec58c253c910/5951950b218e8069e6e63b79/ce873e51a69294a1f12fa0dfb0ce563e/image.png)

The input to this algorithm is a matrix of shape (1, m), where m is the number of input units. This can also be expressed as above as a single-element vector containing a vector with m elements.

## The Input Weight Matrix

![The Input Weight Matrix](https://trello-attachments.s3.amazonaws.com/5741d22c3d23ec58c253c910/5951950b218e8069e6e63b79/2ef0c3df6b4da69598c992ffcfc36a12/image.png)

In order to transform the input matrix of shape (1, m) a proper state matrix of shape (1, n, d), where n is the number of output units, and d is the number of intermediate layers, a one dimensional dot product is required along the last axis of the input matrix, and the first axis of the input weight matrix of shape (m, n, d).

## The State Matrix

![The State Matrix](https://trello-attachments.s3.amazonaws.com/5741d22c3d23ec58c253c910/5951950b218e8069e6e63b79/26061157ed11b331ffac96306b4a9a04/image.png)

The state matrix contains the temporal information learned by the algorithm. Inputs from different points in time combine within the state and form unique activation patterns. The depth of the state determines the fixed temporal window this algorithm can learn.

### Updating The State Matrix

![Updating The State Matrix](https://trello-attachments.s3.amazonaws.com/5741d22c3d23ec58c253c910/5951950b218e8069e6e63b79/93639a386639cade5018d08d4767ef1d/image.png)

At the conclusion of each network forward pass, the state is updated simply by setting equivalence to the current output of the network. This is to say that the new state is the output of this algorithm.

## The State Weight Matrix

![The State Weight Matrix](https://trello-attachments.s3.amazonaws.com/5741d22c3d23ec58c253c910/5951950b218e8069e6e63b79/484fe4e4efd2656070ddea6cba72e597/image.png)

The result of the double contraction between (w_s⊙a) and s must be the same shape as the state matrix (1, n, d). We start with the state weight matrix of shape (n, d, n, d), and see that two dimensions must be reduced. This is accomplished by computing a dot product across the last two dimensions of the state matrix, and the first two dimensions of the state weight matrix of shape, which results in the proper output matrix of shape (1, n, d).

## The Adjacency Matrix

![The Adjacency Matrix](https://trello-attachments.s3.amazonaws.com/5741d22c3d23ec58c253c910/5951950b218e8069e6e63b79/5d2f4343ebd8a185d77c656ef08d10f7/image.png)

In order to compute an update for this algorithm as a single matrix multiplication, an adjacency matrix is used to determine which neurons in each state layer connect to which other neurons in the state. A zero indicates a blocked connection, and a one indicates an allowed connection. Notice the upper right triangle of ones. This is to say that each layer (ℓ + 1) in the state connects to one additional previous layer in the state that layer ℓ in the state.

The expression (w_s⊙a) does not capture that both w_s and a are matrices of different shapes. In order to compute a proper element-wise multiplication, the adjacency matrix must be duplicated along two new axis n times and must be reshaped in order to become the shape of (n, d, n, d) of which the state weight matrix is.

## Calculating Partial Derivatives

These partial derivatives can be calculated from the algorithm expression using the rules of multivariate calculus.

![Input Partial](https://trello-attachments.s3.amazonaws.com/5741d22c3d23ec58c253c910/5951950b218e8069e6e63b79/02239f6e3206e91deff04145e8825746/image.png)

![Input Weight Partial](https://trello-attachments.s3.amazonaws.com/5741d22c3d23ec58c253c910/5951950b218e8069e6e63b79/624c725b28ba2684b7c71ed0ad9980e2/image.png)

![State Partial](https://trello-attachments.s3.amazonaws.com/5741d22c3d23ec58c253c910/5951950b218e8069e6e63b79/ef00405f81022a3434a14b1fbe8b0f96/image.png)

![State Weight Partial](https://trello-attachments.s3.amazonaws.com/5741d22c3d23ec58c253c910/5951950b218e8069e6e63b79/4d5ce19e4961331dfa616354728937fb/image.png)
