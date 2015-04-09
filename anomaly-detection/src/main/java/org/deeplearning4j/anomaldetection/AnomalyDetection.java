package org.deeplearning4j.anomaldetection;

import java.io.IOException;
import java.util.*;

/**
 * Anomaly Detection Test
 *
 * @author  naimtenor
 * Created by ubuntu on 15. 4. 9.
 */
public class AnomalyDetection {

    public static void main(String[] args) throws IOException {
        Map<String, List<Integer>> map = Util.parse(args[0]);

//        System.out.println(map);
        for (Map.Entry<String, List<Integer>> entry : map.entrySet()) {
            System.out.println(entry.getKey() + " = " + entry.getValue());
        }
    }


}
