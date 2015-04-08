package org.deeplearning4j.d3;

import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.IrisDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.LayerFactory;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.override.ClassifierOverride;
import org.deeplearning4j.nn.layers.convolution.ConvolutionDownSampleLayer;
import org.deeplearning4j.nn.layers.convolution.preprocessor.ConvolutionInputPreProcessor;
import org.deeplearning4j.nn.layers.convolution.preprocessor.ConvolutionPostProcessor;
import org.deeplearning4j.nn.layers.factory.LayerFactories;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * @author Adam Gibson
 */
public class ConvnetExample {

    private static Logger log = LoggerFactory.getLogger(ConvnetExample.class);

    public static void main(String[] args) {

        LayerFactory layerFactory = LayerFactories.getFactory(ConvolutionDownSampleLayer.class);
        int batchSize = 110;
        /**
         *
         */
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .optimizationAlgo(OptimizationAlgorithm.LBFGS)
                .dist(Nd4j.getDistributions().createNormal(0, 1))
                .iterations(100).iterationListener(new ScoreIterationListener(1))
                .activationFunction("tanh").filterSize(5, 1, 2, 2).constrainGradientToUnitNorm(true)
                .nIn(4).nOut(3).batchSize(batchSize)
                .layerFactory(layerFactory).dropOut(0.5)
                .list(2)
                .preProcessor(0, new ConvolutionPostProcessor()).inputPreProcessor(0, new ConvolutionInputPreProcessor(2, 2))
                .hiddenLayerSizes(new int[]{9})
                .override(1,new ClassifierOverride(1)).build();

        MultiLayerNetwork network = new MultiLayerNetwork(conf);
        DataSetIterator iter = new IrisDataSetIterator(150, 150);


        org.nd4j.linalg.dataset.DataSet next = iter.next();
        next.normalizeZeroMeanZeroUnitVariance();
        SplitTestAndTrain trainTest = next.splitTestAndTrain(110);
        /**
         * Likely cause: shape[0] mis match on the filter size and the input batch size.
         * Likely need to make a little more general.
         */
        network.fit(trainTest.getTrain());


        //org.nd4j.linalg.dataset.DataSet test = trainTest.getTest();
        Evaluation eval = new Evaluation();
        INDArray output = network.output(trainTest.getTest().getFeatureMatrix());
        eval.eval(trainTest.getTest().getLabels(),output);
        log.info("Score " +eval.stats());
    }

}
