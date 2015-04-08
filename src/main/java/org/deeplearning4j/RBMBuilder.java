package org.deeplearning4j;

import org.deeplearning4j.datasets.iterator.impl.LFWDataSetIterator;
import org.deeplearning4j.models.featuredetectors.rbm.RBM;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.LayerFactory;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.layers.factory.LayerFactories;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ComposableIterationListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.plot.iterationlistener.NeuralNetPlotterIterationListener;
import org.nd4j.linalg.api.rng.distribution.Distribution;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.lossfunctions.LossFunctions;

/**
 * Created by ubuntu on 15. 4. 7.
 */
public class RBMBuilder {

    public static void main(String[] args) {

        /* Load your data. We'll be using the LFW Dataset Iterator for this exercise. */
        // batch size, number of example, image height, image width
        LFWDataSetIterator iter = new LFWDataSetIterator(10, 100, 28, 28);



        /* Create an RBM layer factory. */
        LayerFactory layerFactory = LayerFactories.getFactory(RBM.class);

        /* Set up a NeuralNetConfiguration. */
        NeuralNetConfiguration config = new NeuralNetConfiguration.Builder()
                .layerFactory(layerFactory)
                .nIn(784)
                .nOut(600)
                .applySparsity(true)
                .sparsity(0.1)
                .learningRate(1e-1f)
                .momentum(0.5)
//                .momentumAfter()
                .useAdaGrad(true)
                .resetAdaGradIterations(1)
                .iterations(5)
                .l2(2e-4)
                .iterationListener(new ComposableIterationListener(new NeuralNetPlotterIterationListener(1), new ScoreIterationListener(1)))
//                .dropOut()
                .activationFunction("tanh")
                .numLineSearchIterations(10)
                .weightInit(WeightInit.DISTRIBUTION)
                .optimizationAlgo(OptimizationAlgorithm.GRADIENT_DESCENT)
                .lossFunction(LossFunctions.LossFunction.MSE)
                .build();

        Layer l = LayerFactories.getFactory(RBM.class).create(config);


        /* Normalize the data. */
//        DataSet d = iter.next();
//        d.normalizeZeroMeanZeroUnitVariance();
        int i = 1;
        while (iter.hasNext()) {
            System.out.println("----------------------------- " + i);
            DataSet d = iter.next();
            System.out.println(d);
            d.normalizeZeroMeanZeroUnitVariance();
            l.fit(d.getFeatureMatrix());
            i++;
        }
    }

}
