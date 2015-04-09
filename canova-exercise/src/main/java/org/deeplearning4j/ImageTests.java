package org.deeplearning4j;

import org.canova.api.records.reader.RecordReader;
import org.canova.api.records.reader.impl.ComposableRecordReader;
import org.canova.api.split.FileSplit;
import org.canova.image.recordreader.ImageRecordReader;

import java.io.File;
import java.io.IOException;

/**
 * Image Read Test<br>
 *
 * @author naimtenor
 *
 */
public class ImageTests
{
    public static void main( String[] args ) throws IOException, InterruptedException {

        // arhive util.. not using in this line.
//        ArchiveUtils.unzipFileTo();

        // read image...
        /*
            ~/lfw/[foldername] -> ... matrix transform -> [1, 2, 3]
         */
        RecordReader reader = new ImageRecordReader(28, 28, true);
        reader.initialize(new FileSplit(new File(System.getProperty("user.home"), "lfw")));
//        Collection<Writable> record = reader.next();

//        RecordWriter writer = new CSVRecordWriter(new File("test.csv"));
//        writer.write(record);

//        System.out.println(record);

        RecordReader reader2 = new ImageRecordReader(28, 28);
        reader2.initialize(new FileSplit(new File(System.getProperty("user.home", "MNIST"))));

        RecordReader composable = new ComposableRecordReader(reader, reader2);
        while(composable.hasNext()) {
            System.out.println(composable.next().size());
        }

//        while (reader.hasNext()) {
//            System.out.println(reader.next());
//        }


    }
}
