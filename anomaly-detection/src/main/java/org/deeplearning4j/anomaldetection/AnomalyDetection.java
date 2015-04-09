package org.deeplearning4j.anomaldetection;

import org.canova.api.records.reader.RecordReader;
import org.canova.api.split.FileSplit;

import java.io.File;
import java.io.IOException;
import java.util.*;

/**
 * Anomaly Detection Test
 *
 * @author  naimtenor
 * Created by ubuntu on 15. 4. 9.
 */
public class AnomalyDetection {

    public static void main(String[] args) throws IOException, InterruptedException {
        Map<String, List<Integer>> map = Util.parse(args[0]);

        RecordReader reader = new AnomalyDetectionRecordReader(map, 50, 50);
        reader.initialize(new FileSplit(new File(System.getProperty("user.home"), "/dl4j/data/UCSD_Anomaly_Dataset.v1p2/UCSDped1")));

        while (reader.hasNext()) {
            System.out.println(reader.next());
        }

//        System.out.println(map);
//        for (Map.Entry<String, List<Integer>> entry : map.entrySet()) {
//            System.out.println(entry.getKey() + " = " + entry.getValue());
//        }
    }


}
