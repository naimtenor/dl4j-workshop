# *Day 2: Introduction to Neural Networks*

#### What are we doing today?
Today, we'll be learning about neural networks. We'll learn how to configure a neural net, using a wide range of hyperparameters.

#### Resources 
1. Deep Learning: A Practitioner's Approach (Chapters 1, 2, and 5) 

##**Introduction**

#### What is a neural network?
Artificial neural networks are a class of semi-autonomous algorithms, loosely inspired by the human brain, that learn to classify and reconstruct various data types by repeatedly attempting to minimize the error in their own calculations. 

Hyperparameters can be adjusted to tune neural nets and improve not just their capacity to classify but also to learn. When you're building neural nets, tuning these hyperparameters is essential to getting the results you want. We're going to cover all of the hyperparameters that can be tuned with Deeplearning4j.

##**RBMs**
Restricted Boltzmann machines (RBMs) are two-layer neural nets used to reconstruct and classify data, automatically extract features and reduce dimensionality. RBMs are algorithms that model probability. They are feedforward networks, as data is fed through them in one direction with two biases, rather than one in traditional backpropagation feedforward nets. 

Each RBM has two layers. That is, RBMs are shallow nets. They can be stacked on top of each other to form deep nets of many more layers. Deep nets made of RBMs are called deep belief networks, or DBNs. 

The first layer of each RBM is called the visible layer; the second is called the hidden layer. The visible layer is where the RBM receives input, or raw features; the hidden layer is where RBMs combine input into more complex features. 

When stacked upon each other, the hidden (second) layer of an initial RBM becomes the visible (first) layer of the next RBM. Those stacks of RBMs are called deep-belief networks (DBNs).

1. Create a class in src/main/java called RBMBuilder. 

2. Load your data. We'll be using the LFW Dataset Iterator for this exercise.  

                LFWDataSetIterator iter = new LFWDataSetIterator(10,10,28,28);

3. Normalize the data.  

                DataSet d = iter.next();
                d.normalizeZeroMeanZeroUnitVariance();

4. Create an RBM layer factory.  

                LayerFactory layerFactory = LayerFactories.getFactory(RBM.class);  

5. Set up a NeuralNetConfiguration.  

        NeuralNetConfiguration config = new NeuralNetConfiguration.Builder()


###**Sparsity**
The sparsity hyperparameter recognizes that for some inputs x, only a few features are relevant (or even present). A code or vector representation is said to be sparse when the input activates very few nodes. A sparse representation counts just a few activations across many nodes. If we imagine activations as lit nodes, sparse representations are like the lights of scattered farm houses in a dark landscape.  

1. Add the sparsity parameter to your configuration, and build.  

2. Use the applySparsity parameter to alternate between applying and not applying sparsity. Compare the outcomes of both.

###**Magnitude**
The magnitude of the gradient involves learning rate and momentum, which determine how large of a step we should take relative to our gradient. This can either be an individual learning rate with some decay over time; momentum (which is a complementary term to the learning rate for achieving faster convergence); or adaptive, where there is a learning rate specific to each feature. 

###**Learning Rate**
The learning rate in machine learning is how fast we change the parameter vector as we move through search space. If the learning rate gets too high, we can move towards our goal faster (least amount of error for the function being evaluated), but we may also take a step so large that we shoot right past the best answer to the problem. If we make our learning rate too small, it may take a lot longer than we’d like for our training process to complete. Learning rates are tricky because they end up being specific to the dataset and to other hyperparameters. This creates a lot of overhead for finding the right setting for hyperparameters.

1. Add the learning rate parameter to your configuration, and build.  

        learningRate()

###**Momentum & Momentum After n Iterations**
Momentum gives us an additional factor to use in controlling how fast our learning process converges on its solution. We can speed our training up by increasing momentum, but we may lower our model’s accuracy. Momentum is a factor between 0.0 and 1.0 that is applied to the change/learning rate of the weights over time. Momentum is metaphorically a ball rolling down a surface. 
The ball goes faster as it travels down a slope, or gradient, and quickly across long flat areas, but will slow down when it begins to climb a slope on the other side of a “ravine”. In this way, we can use a type of velocity in how our learning is “moving” through search space to help aid our training process in navigating tricky areas of search space. Over time, we want this velocity to decay to let the training process slow down and converge. Most implementations, such as DL4J, take care of this velocity decay for you.

At the beginning of training, you should set the momentum value to 0.5, given that random initial parameter values can put you in odd starting places in the search space. Over time, a good implementation will increase the momentum towards 0.9. As the reconstruction error rises, or we near a terminating condition, the momentum will begin to fall towards 0.0. 

1. Add the momentum parameter to your neural net configuration. 

        momentum() 

2. Add the momentum after n iterations parameter to your neural net configuration.  

        momentumAfter()


###**AdaGrad & Reset AdaGrad Iterations**
Adagrad is a hyperparameter that helps find the “right” learning rate. With adagrad, the system keeps a custom learning rate *for each parameter* that is adaptive over time, based on changes in learning. Adagrad is the square root of the sum of squares of the historical, component-wise gradient. Adagrad speeds training in the beginning, and slows it as convergence nears, which smoothes training. 

1. Add the AdaGrad parameter to your configuration, and build. It can either be set to true or false (default is true).  

        useAdaGrad()

2. Add the parameter for resetting AdaGrad historical gradient after n iterations, and build.  

        resetAdaGradIterations()

###**Number of Iterations**
In training, an iteration is defined as a single update to the parameter vector. 

1. Add the number of iterations parameter to your configuration, and build. The default is set to 1000, but try a few different values to see how this affects the outcome of your neural net.  

        iterations()

###**Regularization: L2**
Regularization hyperparameters modify the gradient such that it doesn’t step in a direction that leads to overfitting.   

L2 regularization
* is computationally efficient due to having analytical solutions
* has non-sparse outputs
* requires no feature selection  

1. Add the L2 regularization parameter to your configuration, and build.   

        l2()


###**Regularization: Dropout**
Dropout mitigates overfitting in deep learning networks. It also speeds up training. Dropout omits, or mutes, a unit on a hidden layer randomly. It drops "neurons" stochastically so that they don't contribute to the forward pass and back propagation. The forces the net to find other signals within the data, which makes it generalize better.

1. Add the dropout regularization parameter to your configuration, and build.  

        dropOut()


###**Number of Line Search Iterations**
Line search is for finding the direction to go in; it's for finding the optimal line. You try a number of different directions until you find the best one. The default is 100.  

1. Add the number of line search iterations hyperparameter to your configuration, and build. Try out different values, and compare the results.  

        numLineSearchIterations()


###**Weight Initialization Scheme**
Weights are usually initialized to small random values chosen from a zero-mean Gaussian with a standard deviation of 0.01. Larger random values can speed up learning, but they may lead to slightly worse results. 

Options for the weight initialization parameter:

    * VI: Variance normalized initialization (Glorot)
    * SI: Sparse initialization (Martens)
    * ZERO: straight zeros (for logistic regression)
    * DISTRIBUTION: Sample weights from a distribution
    * NORMALIZED: Normalized weight initialization
    * UNIFORM: Sample weights from uniform distribution


1. Add the weight initialization scheme parameter to your configuration, and build.  

        weightInit()

2. Add the weight initialization distribution parameter to your configuration, and build.  

        dist()

###**Optimization Algorithm**
Quick calculus review: first-order derivatives indicate the rate of change at any given point on a curve. If you start by tracing a line that charts change in position over time, then your first-order derivative is velocity, the rate of change of position. In this situation, the second-order derivative is acceleration, or the rate of change of velocity (i.e. speeding up or slowing down).  

Optimization algorithms fall into two camps: first-order and second-order.  

**First-order optimization algorithms** calculate the Jacobian matrix. The Jacobian is a matrix of partial derivatives that calculate the degree to which changing parameters changes the output. You have one partial derivative per parameter (to calculate partial derivatives, all other variables are momentarily treated as constants). The algorithm then takes one step in the direction specified by the Jacobian and recalculates an error. 

The Jacobian can be defined as a differentiable function, which approximates a given point x (a number in the parameter vector). This is the direct approximation, mapping a given input onto F, which is our function.  

If we think about taking one step at a time to reach an objective, then first-order methods calculate a gradient (Jacobian) at each step to determine which direction to go next. That means at each iteration, or step, we are trying to find the next best possible direction to go, as defined by our objective function. This is why we consider optimization algorithms to be a “search.” They are finding a path toward minimal error.  

**Gradient descent** is a member of this path-finding class of algorithms. Variations of gradient descent exist, but at its core, it finds the next step in the right direction with respect to an objective at each iteration. Those steps moves us toward a global minimum error or maximum likelihood.  


**Second-order algorithms** calculate the derivative of the Jacobian (i.e., the derivative of a matrix of derivatives), which is called the Hessian. Second-order methods are used for controlling the learning rate (aka step size), which is how much an algorithm adjusts the weights as it tries to learn. They’re also useful in determining which way your gradient is sloping.  

Second-order methods include L-BFGS, conjugate gradient, and Hessian-free. We will cover these in depth in Chapter 5. For now, think of every one of these as a black-box search algorithm that determines the best way to minimize error, given an objective and a defined gradient relative to each layer.  

**Limited-memory BFGS (L-BFGS)** is a mathematical optimization algorithm and a so-called quasi-newton method. It’s a variation of the Broyden-Fletcher-Goldfarb-Shanno (BFGS) algorithm, named after its many inventors, which limits the amount of memory.  

L-BFGS approximates the inverse Hessian matrix to direct the search towards more promising areas of parameter space. Where BFGS stores the full n x n inverse, Hessian L-BFGS contrasts only stores a few vectors that represent an approximation of it.  L-BFGS performs faster because it uses approximated second-order information (modeling the interaction between variables). L-BFGS and Conjugate Gradient in practice can be faster and more stable than SGD methods.  

**Conjugate Gradient** guides the direction of the line search process based on conjugacy information. Conjugate gradient methods are considered Lanczos methods and move through Krylov space. Conjugate gradient methods focus on minimizing the conjugate l2 norm. Conjugate gradient is very similar to gradient descent in that it performs line search. The major difference is that Conjugate gradient requires each successive step in the line search process be conjugate to one another with respect to direction.  

**Hessian-free optimization** is related to Newton’s method, but better minimizes the quadratic function we get. It is a powerful optimization method adapted to neural networks by James Martens in 2010. We find the minimum of the quadratic function with an iterative method called conjugate gradient.

Optimization algorithm options:  

    * GRADIENT_DESCENT
    * CONJUGATE_GRADIENT
    * HESSIAN_FREE
    * LBFGS
    * ITERATION_GRADIENT_DESCENT 

1. Add the optimization algorithm parameter to your configuration, and pass in one of the above options.  

        optimizationAlgo()


###**Loss Function**
A loss or objective function is a way of telling a neural network what it should be optimizing its parameters for. Loss functions are used in machine learning to determine the penalty for an incorrect classification of an input vector. Think of it like a compass that leads the optimization algorithm down a hill towards an optimal solution.  

Options for loss function parameter:  

     * MSE: Mean Squared Error: Linear Regression
     * EXPLL: Exponential log likelihood: Poisson Regression
     * XENT: Cross Entropy: Binary Classification
     * MCXENT: Multiclass Cross Entropy
     * RMSE_XENT: RMSE Cross Entropy
     * SQUARED_LOSS: Squared Loss
     * RECONSUTRCTION_CROSSENTROPY: Reconstruction Cross Entropy
     * NEGATIVELOGLIKELIHOOD: Negative Log Likelihood

1. Add the loss function parameter to your configuration, and pass in one of the above options.  

        lossFunction()


###**Minimize or Maximize Objective**
Depending on whether your loss or objective function should be minimized or maximized, you may want to adjust this parameter. It's a boolean parameter for minimizing the objective. Defaults to false.

1. Add the minimize objective parameter to your configuration, and build.  

        minimize()


###**Concatenate Hidden Biases or Add it**
You can either add the biases or concatenate them into one matrix when you're doing the calculations. 

1. Add the concatenate hidden biases parameter to your configuration, and build.  

        concatBiases() 


###**Constrain the Gradient to Unit Norm**
After you calculate the Jacobian at each iteration, you divide by the norm of the gradient.

1. Add the constrain gradient to unit norm parameter to your configuration, and build.  

        constrainGradientToUnitNorm()


###**Random Seed for Sampling**
You can either choose a random seed for sampling or implement your own random number generator.  

1. Add the seed parameter to your configuration, and build.  

        seed()

2. If you want to implement your own random number generator, add the rng parameter to your configuration, and build.  

        rng()


###**Listeners**
Iteration Listener (e.g., print every x iterations). Listeners can be used for debugging.  

1. Add the listener parameter to your configuration, and build.  

        iterationListener()


###**Step Function**
This is a custom step function for line search. You can do it relative to line search or just a simple update. You can also do inverse.

1. Add the step function parameter to your configuration, and build.  

        stepFunction()

###**Layer Factory**
Specify the type of neural network.

1. Add the layer factory parameter to your configuration, and build.  

        layerFactory()

###**Variables**
These are parameters for the model.

1. Add the variables parameter to your configuration, and build.  

        variables()

###**Feedforward Nets: nIn and nOut**
Specify the number of nodes for input layer and output layer. The number of nodes for the first layer is always equal to the number of features.  

1. Add the nIn parameter to your configuration, and build.  

        nIn()

2. Add the nOut parameter to your configuration, and build.  

        nOut()

###**Activation Function**
Activation functions define the mapping of the input to the output via a nonlinear transform function for units in neural networks. Examples of these functions are sigmoid and tanh. 

Activation functions are the nonlinear transfer functions that describe how to transfer the data from one layer to another (or in the case of the output layer: how to contain the input in a limited space). These nonlinear transforms typically take a function and map it onto a constrained space that looks like a modified version of the sigmoid function that we saw in Chapter 1.
Activation functions also contribute to the “nonlinear” part of the equation. A nonlinear transform over the data allows the network to learn complex patterns that might not otherwise be possible. The nonlinear part is very similar to an SVM’s kernel function. The kernel’s job in an SVM is to reproject the data onto a new space. This reprojection allows an SVM to build better decision boundaries in the data. These decision boundaries are what give the SVM its power for learning noisier functions. 
An activation function in the context of deep learning is used for mapping a raw activation (x * weights + bias for a feedforward network) to a nonlinear, constrained, space where the network can learn patterns. Activation functions are applicable for all objective functions. Depending on the activation function you pick, you will find that some objective functions are more appropriate for different kinds of data (say, dense vs. sparse).



1. Add the activation function parameter to your configuration, and build.  

        activationFunction()

###**RBMs: Visible Unit, Hidden Unit, and k**
RBMs: A standard RBM has a visible layer of units and a hidden layer of units. The wiring of RBMs is setup such that every visible unit is connected to every other hidden unit yet no units from the same layer are connected. We consider the visible units to be observable in that they take training vectors as input yet the hidden units to be “feature detectors” learned from said input. We can also use the visible units to show us what the learned representation is based on a given input.  

Visible units: Number of nodes in the visible layer

Hidden units: Number of nodes in the hidden layer

k: Number of iterations for contrastive divergence


1. Add the visible unit parameter to your configuration, and build.  

        visibleUnit()

2. Add the hidden unit parameter to your configuration, and build.  

        hiddenUnit()

3. Add the k parameter to your configuration, and build.  

        k()


###**Weight Shape**
This is typically the number of ins (input nodes) and number of outs (output nodes). However, it can be any size.

1. Add the weight shape parameter to your configuration, and build.  

        weightShape()

##**Create the RBM**

1. Using layerFactory.create(), create and fit an RBM.  

                RBM rbm = layerFactory.create(conf);
                rbm.fit(d.getFeatureMatrix());


##**Autoencoders**
1. Load the data. We'll be using the MNIST dataset for this exercise.  

                MnistDataFetcher fetcher = new MnistDataFetcher(true);

2. Create an autoencoder layer factory.  

                LayerFactory layerFactory = LayerFactories.getFactory(AutoEncoder.class);

3. Build a neural net configuration using this layer factory. Make sure to add an iteration listener for this exercise, so you can see the neural network's progress visually.  

                .iterationListener(new ComposableIterationListener(new NeuralNetPlotterIterationListener(1),new ScoreIterationListener(1)))

4. Fetch data from the MNIST data fetcher. Get the feature matrix from this data.   

        fetcher.fetch(100);
        DataSet d2 = fetcher.next();
        INDArray input = d2.getFeatureMatrix();

5. Build your autoencoder, and fit.  

        AutoEncoder da = layerFactory.create(conf);
        da.setParams(da.params());
        da.fit(input);


###**Corruption Level**
Corrupts the given input by doing a binomial sampling given the corruption level. This is similar to dropout. However, you add random Gaussian noise, rather than zeroing it out.

1. Add the corruption level parameter to your autoencoder configuration, and build again. The default is set to 0.3, but try a few different values between 0 and 1 to see how this affects the outcome.  

        corruptionLevel()


##**Multilayer Neural Networks**
Now that we know how to configure a neural network, let's combine multiple layers to build a more complex neural net. 

###**List Function**
Using the same NeuralNetConfiguration from earlier, you can easily configure multiple layers at the same time. The list function serves this purpose.  It will replicate your configuration n times and build a layerwise configuration.

1. Create a new class in src/main/java. Name this class MultiLayerBuilder.  

2. Copy the main() method from your RBM configuration into this class.

3. Change the configuration to a MultiLayerConfiguration.  

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()

4. Add the list parameter to the neural net configuration, and specify the number of layers you want to build with the given parameters.  

        list(n)

5. Instead of building an RBM from your layer factory, build a multilayernetwork.  

        MultiLayerNetwork network = new MultiLayerNetwork(conf);


###**Hidden Layers**
1. Add the hidden layer sizes parameter to your MultiLayerConfiguration, and specify the sizes of your hidden layers.  

        hiddenLayerSizes()

###**Drop Connect**
Drop connect is a generalization of dropout. A randomly selected subset of weights within the neural network are set to zero. 

1. Add the drop connect parameter to your configuration, and specify whether or not you want to use drop connect.  

        useDropConnect()

2. Normalize your data, and fit.  

        org.nd4j.linalg.dataset.DataSet next = iter.next();
        next.normalizeZeroMeanZeroUnitVariance();
        network.fit(next);


###**Override**
When you're building a multilayer network, you won't necessarily want the same configuration for all layers. In that case, you should use the override method to modify any configurations that are necessary.  

1. Use override(), and specify the layer you want to modify, as well as the builder to override values for. You'll likely want to change the loss function for the last layer, as well as the activation function.


