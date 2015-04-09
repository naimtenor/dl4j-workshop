package org.deeplearning4j.anomaldetection;

import org.apache.commons.io.FileUtils;
import org.nd4j.linalg.util.ArrayUtil;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Util class for AnomalyDetection
 *
 * @author naimtenor
 * Created by ubuntu on 15. 4. 9.
 */
public class Util {

    /**
     * Parse Map data from String data.
     *
     * @param arg
     * @return
     * @throws IOException
     */
    public static Map<String, List<Integer>> parse(String arg) throws IOException {
        // read line from file
        List<String> lines = FileUtils.readLines(new File(arg));
        Map<String, List<Integer>> map = new HashMap<>();

        for (int i = 0 ; i < lines.size() ; i++) {
            String[] split = lines.get(i).split(" = ");
            if (i > 0) {
                Range range = new Range(split[1]);
                List<Integer> rangeList = new ArrayList<>();

                for (int j = 0 ; j < range.getBegins().length ; j++) {
                    int[] range3 = ArrayUtil.range(range.getBegins()[j], range.getEnds()[j]);
                    for (int curr : range3) {
                        rangeList.add(curr);
                    }
                }
                String fileName = i > 9 ? "Test0" + i : "Test00" + i;
                map.put(fileName, rangeList);
            }
        }

        return map;
    }

}
