package org.deeplearning4j.d3;

import org.deeplearning4j.models.classifiers.lstm.LSTM;
import org.deeplearning4j.nn.api.LayerFactory;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.layers.factory.LayerFactories;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.util.FeatureUtil;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * @author Adam Gibson
 */
public class RecurrentNetExample {

    private static Logger log = LoggerFactory.getLogger(RecurrentNetExample.class);

    public static void main(String[] args) {
        LayerFactory factory = LayerFactories.getFactory(LSTM.class);

        NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder().activationFunction("tanh")
                .layerFactory(factory).optimizationAlgo(OptimizationAlgorithm.ITERATION_GRADIENT_DESCENT)
                .lossFunction(LossFunctions.LossFunction.RECONSTRUCTION_CROSSENTROPY).iterationListener(new ScoreIterationListener(10))
                .nIn(4).nOut(4).build();
        LSTM l = factory.create(conf);
        INDArray predict = FeatureUtil.toOutcomeMatrix(new int[]{0,1,2,3},4);
        l.fit(predict);
        INDArray out = l.activate(predict);
        log.info("Out " + out);
    }

}
