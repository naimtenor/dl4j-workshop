package org.deeplearning4j.anomaldetection;

import org.canova.api.io.data.IntWritable;
import org.canova.api.records.reader.RecordReader;
import org.canova.api.split.FileSplit;
import org.canova.api.split.InputSplit;
import org.canova.api.writable.Writable;
import org.canova.common.RecordConverter;
import org.canova.image.loader.ImageLoader;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.File;
import java.io.IOException;
import java.net.URI;
import java.util.*;

/**
 * Custom record reader for
 * Anomaly Detection Record reader<br>
 *     implements RecordReader
 *
 * @author naimtenor
 * Created by ubuntu on 15. 4. 9.
 */
public class AnomalyDetectionRecordReader implements RecordReader {

    private static final String DIRECTORY_NAME_GT = "_gt";

    private Iterator<File> data;

    private Map<String, Boolean> label = new HashMap<>();

    private Map<String, List<Integer>> anomailies;

    private ImageLoader loader;

    /**
     * defalut constructor
     * using root directory
     * and anomaily list..
     *
     * @param anomailies Anomal data lists
     */
    public AnomalyDetectionRecordReader(Map<String, List<Integer>> anomailies, int rows, int cols) {
        this.anomailies = anomailies;
        loader = new ImageLoader(rows, cols);
    }

    @Override
    public void initialize(InputSplit split) throws IOException, InterruptedException {
        if (split instanceof FileSplit) {
            FileSplit fileSplit = (FileSplit) split;
            URI[] u = fileSplit.locations();
            List<File> data = new ArrayList<>();

            for (URI file : u) {
                File inputFile = new File(file);
                if (!inputFile.exists()) {
                    throw new IllegalStateException("Unable to find file " + inputFile.getAbsolutePath());
                }
                if (inputFile.isDirectory()) {

                } else {
                    File video = inputFile.getParentFile();
                    String videoName = video.getName();
                    // if directory name not containe _gt...
                    if (!videoName.contains(DIRECTORY_NAME_GT)) {
                        String fileName = inputFile.getName();
                        try {
                            int clip = Integer.parseInt(fileName.replaceAll(".tif", ""));
                            boolean anomaly = !anomailies.containsKey(videoName) || anomailies.get(videoName).contains(clip);
                            label.put(fileName, anomaly);
                            data.add(inputFile);
                        } catch (NumberFormatException e) {
                            continue;
                        }
                    }
                }
            }
            this.data = data.iterator();
        }
    }

    @Override
    public Collection<Writable> next() {
        File next = data.next();
        try {
            INDArray row = loader.asRowVector(next);
            Collection<Writable> record = RecordConverter.toRecord(row);
            record.add(new IntWritable(label.get(next.getName()) ? 1 : 0));
            return record;
        } catch (Exception e) {
            e.printStackTrace();
        }
        return null;
    }

    @Override
    public boolean hasNext() {
        return data.hasNext();
    }

    @Override
    public void close() throws IOException {

    }
}
