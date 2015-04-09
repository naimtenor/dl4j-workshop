package org.deeplearning4j.anomaldetection;

import org.canova.api.records.reader.RecordReader;
import org.canova.api.split.FileSplit;
import org.deeplearning4j.datasets.canova.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.models.featuredetectors.rbm.RBM;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.override.ClassifierOverride;
import org.deeplearning4j.nn.conf.preprocessor.ZeroMeanAndUnitVariancePrePreProcessor;
import org.deeplearning4j.nn.layers.factory.LayerFactories;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.SerializationUtils;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.io.IOException;
import java.util.List;
import java.util.Map;

import static java.lang.Integer.MAX_VALUE;

/**
 * Anomaly Detection Test
 *
 * @author  naimtenor
 * Created by ubuntu on 15. 4. 9.
 */
public class AnomalyDetectionDBN {

    public static void main(String[] args) throws IOException, InterruptedException {
        Map<String, List<Integer>> map = Util.parse(args[0]);

        int rows = 50;
        int cols = 50;

        if (!new File("dataset.ser").exists()) {
            RecordReader reader = new AnomalyDetectionRecordReader(map, rows, cols);
            reader.initialize(new FileSplit(new File(System.getProperty("user.home"), "/dl4j/data/UCSD_Anomaly_Dataset.v1p2/UCSDped1")));
            DataSetIterator iter = new RecordReaderDataSetIterator(reader, MAX_VALUE, rows * cols, 2);
            SerializationUtils.saveObject(iter.next(), new File("dataset.ser"));
        }
        DataSet next = SerializationUtils.readObject(new File("dataset.ser"));
        // normal is zero
        // anomal is 1
//        while (iter.hasNext()) {
//            DataSet next = iter.next();
//            next.normalizeZeroMeanZeroUnitVariance();
//            next.filterAndStrip(new int[]{1});
//            DataSet normal = next.filterBy(new int[]{1});
//        }

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .layerFactory(LayerFactories.getFactory(RBM.class))
                .constrainGradientToUnitNorm(true)
                .iterationListener(new ScoreIterationListener(1))
                .learningRate(1e-1)
                .visibleUnit(RBM.VisibleUnit.GAUSSIAN)
                .hiddenUnit(RBM.HiddenUnit.RECTIFIED)
                .weightInit(WeightInit.DISTRIBUTION)
                .dist(Nd4j.getDistributions().createNormal(1e-5, 1))
                .activationFunction("tanh")
                .lossFunction(LossFunctions.LossFunction.RMSE_XENT)
                .nIn(rows * cols)
                .nOut((int) (rows * cols * 0.75))
                .list(4)
                .override(3, new ClassifierOverride(3))
                .hiddenLayerSizes((int) (rows * cols * 0.75), (int) (rows * cols * 0.75), (int) (rows * cols * 0.75))
                .inputPreProcessor(0, new ZeroMeanAndUnitVariancePrePreProcessor())
                .build();


        System.out.println(next.numExamples());

        MultiLayerNetwork network = new MultiLayerNetwork(conf);
        Layer transporsed = network.getOutputLayer().transpose();
        network.fit(next);
    }


}
