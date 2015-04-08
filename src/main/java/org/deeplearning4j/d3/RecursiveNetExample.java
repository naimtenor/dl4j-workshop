package org.deeplearning4j.d3;

import org.deeplearning4j.datasets.fetchers.MnistDataFetcher;
import org.deeplearning4j.models.featuredetectors.autoencoder.recursive.RecursiveAutoEncoder;
import org.deeplearning4j.nn.api.LayerFactory;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.layers.factory.LayerFactories;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.lossfunctions.LossFunctions;

/**
 * @author Adam Gibson
 */
public class RecursiveNetExample {

    public static void main(String[] args) throws Exception {
        MnistDataFetcher fetcher = new MnistDataFetcher(true);
        LayerFactory layerFactory = LayerFactories.getFactory(RecursiveAutoEncoder.class);
        NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder()
                .momentum(0.9f)
                .optimizationAlgo(OptimizationAlgorithm.ITERATION_GRADIENT_DESCENT)
                .corruptionLevel(0.3).weightInit(WeightInit.VI)
                .iterations(100).iterationListener(new ScoreIterationListener(10))
                .lossFunction(LossFunctions.LossFunction.RECONSTRUCTION_CROSSENTROPY)
                .learningRate(1e-1f).nIn(784).nOut(600).layerFactory(layerFactory).build();

        fetcher.fetch(10);
        DataSet d2 = fetcher.next();

        INDArray input = d2.getFeatureMatrix();

        RecursiveAutoEncoder da = layerFactory.create(conf);
        da.setParams(da.params());
        da.fit(input);
    }


}
