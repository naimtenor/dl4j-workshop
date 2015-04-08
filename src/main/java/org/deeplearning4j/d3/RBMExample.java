package org.deeplearning4j.d3;

import org.deeplearning4j.datasets.iterator.impl.LFWDataSetIterator;
import org.deeplearning4j.models.featuredetectors.rbm.RBM;
import org.deeplearning4j.nn.api.LayerFactory;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.layers.factory.LayerFactories;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.lossfunctions.LossFunctions;

/**
 * @author Adam Gibson
 */
public class RBMExample {
    public static void main(String[] args) {
        LFWDataSetIterator iter = new LFWDataSetIterator(10,10,28,28);
        DataSet d = iter.next();

        d.normalizeZeroMeanZeroUnitVariance();

        int nOut = 600;
        LayerFactory layerFactory = LayerFactories.getFactory(RBM.class);

        NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder()
                .hiddenUnit(RBM.HiddenUnit.RECTIFIED).weightInit(WeightInit.VI)
                .iterationListener(new ScoreIterationListener(1))
                .visibleUnit(RBM.VisibleUnit.GAUSSIAN).layerFactory(layerFactory)
                .optimizationAlgo(OptimizationAlgorithm.ITERATION_GRADIENT_DESCENT)
                .lossFunction(LossFunctions.LossFunction.RMSE_XENT)
                .learningRate(1e-3f)
                .nIn(d.numInputs()).nOut(nOut).layerFactory(layerFactory).build();

        RBM rbm = layerFactory.create(conf);

        rbm.fit(d.getFeatureMatrix());
    }


}
