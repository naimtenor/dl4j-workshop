package org.deeplearning4j.anomaldetection;

import org.canova.api.records.reader.RecordReader;
import org.canova.api.split.InputSplit;
import org.canova.api.writable.Writable;

import java.io.IOException;
import java.util.Collection;

/**
 * Anomaly Detection Record reader<br>
 *     implements RecordReader
 *
 * @author naimtenor
 * Created by ubuntu on 15. 4. 9.
 */
public class AnomalyDetectionRecordReader implements RecordReader {

    @Override
    public void close() throws IOException {

    }

    @Override
    public void initialize(InputSplit split) throws IOException, InterruptedException {

    }

    @Override
    public Collection<Writable> next() {
        return null;
    }

    @Override
    public boolean hasNext() {
        return false;
    }
}
