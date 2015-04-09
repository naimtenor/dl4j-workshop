package org.deeplearning4j;

import org.canova.api.records.reader.RecordReader;
import org.canova.api.records.reader.impl.ComposableRecordReader;
import org.canova.api.split.FileSplit;
import org.canova.image.recordreader.ImageRecordReader;
import org.canova.sound.recordreader.WavFileRecordReader;

import java.io.File;
import java.io.IOException;

/**
 * Wav Read Test and Image Read Test <br>
 *     ComposableReader Test too....
 *
 * @author naimtenor
 */
public class BirdCalls {

    public static void main(String[] args) throws IOException, InterruptedException {
        RecordReader wavReader = new WavFileRecordReader(true);
        wavReader.initialize(new FileSplit(new File(System.getProperty("user.home"), "/dl4j/data/mlsp_contest_dataset/essential_data/src_wavs")));

        RecordReader imageReader = new ImageRecordReader(28, 28, true);
        imageReader.initialize(new FileSplit(new File(System.getProperty("user.home"), "/dl4j/data/mlsp_contest_dataset/supplemental_data/filtered_spectrograms_jpg")));

        RecordReader composable = new ComposableRecordReader(wavReader, imageReader);
        while (composable.hasNext()) {
            System.out.println(composable.next().size());
        }

//        while (wavReader.hasNext()) {
//            System.out.println(imageReader.next());
//        }
    }
}
