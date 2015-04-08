package org.deeplearning4j;

import org.deeplearning4j.d3.ConvnetExample;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.IrisDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.models.featuredetectors.autoencoder.AutoEncoder;
import org.deeplearning4j.models.featuredetectors.rbm.RBM;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.override.ClassifierOverride;
import org.deeplearning4j.nn.conf.override.ComposableOverride;
import org.deeplearning4j.nn.conf.override.ConfOverride;
import org.deeplearning4j.nn.layers.factory.LayerFactories;
import org.deeplearning4j.nn.layers.factory.PretrainLayerFactory;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ComposableIterationListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.plot.iterationlistener.NeuralNetPlotterIterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * @author Adam Gibson
 */
public class DBNExample {
    private static Logger log = LoggerFactory.getLogger(ConvnetExample.class);

    public static void main(String[] args) throws Exception {
        Nd4j.MAX_SLICES_TO_PRINT = -1;
        Nd4j.MAX_ELEMENTS_PER_SLICE = -1;
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .iterations(100).layerFactory(new PretrainLayerFactory(RBM.class))
                .iterationListener(new ComposableIterationListener(new ScoreIterationListener(1), new NeuralNetPlotterIterationListener(1)))
                .weightInit(WeightInit.DISTRIBUTION).dist(Nd4j.getDistributions().createUniform(0,1))
                .activationFunction("tanh").momentum(0.9)
                .optimizationAlgo(OptimizationAlgorithm.LBFGS)
                .constrainGradientToUnitNorm(true).k(1).regularization(true).l2(2e-4)
                .visibleUnit(RBM.VisibleUnit.GAUSSIAN).hiddenUnit(RBM.HiddenUnit.RECTIFIED)
                .lossFunction(LossFunctions.LossFunction.RMSE_XENT)
                .learningRate(1e-1f).iterationListener(new ScoreIterationListener(2))
                .nIn(4).nOut(3).list(2)
                .hiddenLayerSizes(new int[]{3})
                .override(new ComposableOverride(new ClassifierOverride(1), new ConfOverride() {
                    @Override
                    public void overrideLayer(int i, NeuralNetConfiguration.Builder builder) {
                        if (i == 1) {
                            builder.layerFactory(LayerFactories.getFactory(AutoEncoder.class));
                            builder.activationFunction("sigmoid");
                            builder.learningRate(1e-3);
                        } else if (i == 2) {
                            builder.learningRate(1e-1);
                        }
                    }
                })).build();

        NeuralNetConfiguration conf2 = new NeuralNetConfiguration.Builder()
                .layerFactory(LayerFactories.getFactory(RBM.class))
                .nIn(784).nOut(600).applySparsity(true).sparsity(0.1)
                .build();

        Layer l = LayerFactories.getFactory(RBM.class).create(conf2);



        MultiLayerNetwork d = new MultiLayerNetwork(conf);


        DataSetIterator iter = new IrisDataSetIterator(150, 150);

        DataSet next = iter.next();

        Nd4j.writeTxt(next.getFeatureMatrix(),"iris.txt","\t");

        next.normalizeZeroMeanZeroUnitVariance();

        SplitTestAndTrain testAndTrain = next.splitTestAndTrain(110);
        DataSet train = testAndTrain.getTrain();

        d.fit(train);




        DataSet test = testAndTrain.getTest();


        Evaluation eval = new Evaluation();
        INDArray output = d.output(test.getFeatureMatrix());
        eval.eval(test.getLabels(),output);
        log.info("Score " + eval.stats());


    }

}
