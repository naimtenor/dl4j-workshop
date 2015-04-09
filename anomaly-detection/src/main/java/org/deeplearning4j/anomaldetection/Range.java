package org.deeplearning4j.anomaldetection;

import java.util.Arrays;

/**
 * Inner class for AnomalyDetection<br>
 *
 * @author naimtenor
 */
public class Range {

    private int[] begins;

    private int[] ends;

    public Range(String toParse) {
        final String delimeter = ", ";
        String[] intervals = null;

        if (toParse.contains(delimeter)) {
            intervals = toParse.split(delimeter);
        } else {
            intervals = new String[]{toParse};
        }

        begins = new int[intervals.length];
        ends = new int[intervals.length];

        for (int i = 0 ; i < intervals.length ; i++) {
            String[] interval = intervals[i].replace("[", "").replace("];", "").split(":");
            begins[i] = Integer.parseInt(interval[0]);
            ends[i] = Integer.parseInt(interval[1]);
        }
    }

    public int[] getBegins() {
        return begins;
    }

    public void setBegins(int[] begins) {
        this.begins = begins;
    }

    public int[] getEnds() {
        return ends;
    }

    public void setEnds(int[] ends) {
        this.ends = ends;
    }

    @Override
    public String toString() {
        return "Range{" +
                "begins=" + Arrays.toString(begins) +
                ", ends=" + Arrays.toString(ends) +
                '}';
    }
}