package org.deeplearning4j;

import org.canova.api.records.reader.RecordReader;
import org.canova.api.split.FileSplit;
import org.canova.api.writable.Writable;
import org.canova.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.iterator.impl.LFWDataSetIterator;
import org.deeplearning4j.models.featuredetectors.rbm.RBM;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.LayerFactory;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.layers.factory.LayerFactories;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.SerializationUtils;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.io.IOException;
import java.util.Collection;

/**
 * Created by ubuntu on 15. 4. 8.
 */
public class RBMExample {

    public static void main(String[] args) throws IOException, InterruptedException {
        RecordReader reader = new ImageRecordReader(28, 28);
        reader.initialize(new FileSplit(new File(System.getProperty("user.home"), "lfw")));
        Collection<Writable> next = reader.next();

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
//                .iterations(40)
                .nIn(d.numInputs()).nOut(nOut).layerFactory(layerFactory).build();

        RBM rbm = layerFactory.create(conf);

        rbm.fit(d.getFeatureMatrix());
        SerializationUtils.saveObject(rbm, new File("myfile.ser"));
        Layer layer = SerializationUtils.readObject(new File("myfile.ser"));

        INDArray params = layer.params();
        String confJson = layer.conf().toJson();

        System.out.println("------------------ confing by json -----------------");
        System.out.println(confJson);
        System.out.println("----------------------------------------------------");
        NeuralNetConfiguration conf3 = NeuralNetConfiguration.fromJson(confJson);
        Layer sameLayer = layerFactory.create(conf3);
        sameLayer.setParams(params);

        Nd4j.writeTxt(params, "somepath.txt", ",");

        INDArray load = Nd4j.readTxt("somepath.txt", ",");
    }
}
