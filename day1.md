#**Day 1: Intro to the Deeplearning4j Ecosystem**
#### What are we doing today?
* We'll be learning Deeplearning4j, Canova, and ND4J. We will be setting up a data pipeline with these tools.

#### Deep learning use cases
* Face/image recognition
* Voice search
* Speech-to-text (transcription)
* Spam filtering (anomaly detection)
* E-commerce fraud detection

#### Why is this important/relevant?
* Automatic Grouping of photos
* Cross platform models (train on the cloud deploy to phone)
* Train models on the phone


#### Resources
* Unit tests
* [Deeplearning4j Website](deeplearning4j.org)
* [ND4J Website](nd4j.org)

##**Introduction**
* Deeplearning4j is the first commercial-grade, open-source, distributed deep-learning library written for Java and Scala.
Integrated with Hadoop and Spark, DL4J is designed to be used in business environments. It aims to be cutting-edge plug and play, more convention than configuration, which allows for fast prototyping.
* ND4J is a scientific computing library for the JVM. It is similar to Numpy and MATLAB.
* Canova is a vectorization tool. It converts raw data into usable vector formats usable with machine learning algorithms.

##**Setup** 
1. Check your version of Java. To test which version of Java you have, type the following into your command line:  

            java -version

    * If you don't have Java 7 installed on your machine, download the [Java Development Kit (JDK) here](http://www.oracle.com/technetwork/java/javase/downloads/jdk7-downloads-1880260.html). 

2. Download [IntelliJ IDE](https://www.jetbrains.com/idea/download/) if you don't already have it. 

3. Download Jblas if you don't already have it. If you're on a Linux machine, follow these steps:

    * Due to our reliance on Jblas for CPUs, native bindings for Blas are required.

            Fedora/RHEL
            yum -y install blas

            Ubuntu
            apt-get install libblas*

    * If you're using IntelliJ as your IDE, this should work already.  

4. Follow [these instructions](http://nd4j.org/getstarted.html#maven) to set up Maven. Here, you will also follow instructions to install ND4J. 

5. Install Deeplearning4j and Canova, using Maven as you did for installing ND4J. You can search for Deeplearning4j and Canova on the [MVNRepository](http://mvnrepository.com/). Select the latest version, and copy the <dependency> code into your pom.xml file.

##**Today's Goal**

Now that we have installed the necessary tools, we can start loading data and getting to know our tools and dataset.  

By the end of the day, we should understand what vectorization is and why we need it. We should know how to vectorize multiple data types and build out data pipelines using the Deeplearning4j ecosystem.

##**What can we do with ND4J?**

ND4J is a scientific computing library with Java and Scala APIs. It can handle linear algebra, calculus, signal processing. It has a versatile NDArray object, or tensor, as its basic data structure.   

**For the following steps, print each array to see how ND4J works.**  

1. Create a new Java class inside of src/main/java. Right click on the java package. Select New -> Java Class. Name it Nd4jExamples.  

2. Create a "main" method inside of your new class.  

3. Create a row vector with the specified number of columns.  

        INDArray arr = Nd4j.create(4);
        System.out.println(arr);

4. Create a row vector with the specified number of columns (all values set to equal 1).  

        INDArray arr2 = Nd4j.ones(4);

5. Create a row vector with 10 columns, ranging from 1 to 10.  

        INDArray arr3 = Nd4j.linspace(1, 10, 10);

6. Add two arrays.  

        arr.add(arr2);

7. Add two arrays in-place. Notice the difference in the method name.  

        arr.addi(arr2);

8. Transpose a matrix.  

        INDArray arrT = arr.transpose();

9. Compute row (1) and column (0) sums.   

        Nd4j.sum(arr4, 1);

        Nd4j.sum(arr4, 0);

10. Check array shape.   

        System.out.println(Arrays.toString(arr2.shape));

11. Assign the value 5 to each element of an array (just like Numpy's "fill" method).  

        arr.assign(5);

12. Reshape the array.   

        arr2 = arr2.reshape(2, 2);

13. Sort the array. Also try sorting *and* returning sorted indices.  

        arr2 = Nd4j.sort(arr2, 0, true);
       
        System.out.println(Arrays.toString(Nd4j.sortWithIndices(arr2, 0, true)));

14. Compute basic statistical properties (mean and standard deviation).  

        Nd4j.mean(arr);
      
        Nd4j.std(arr);

15. Find min and max values.  

        Nd4j.max(arr3);
        
        Nd4j.min(arr3);

16. Boolean indexing: Where a given condition holds true, apply a function to an NDArray.

        * In this example, replace any values below 5 with 5  

        BooleanIndexing.applyWhere(arr3, Conditions.lessThan(5), new Value(5));  

        * In this example, replace any NaN values with 0  

        BooleanIndexing.applyWhere(arr3, Conditions.isNan(), new Value(0));  

        * Here, we can check if we successfully replaced every value less than 5. This should return true.  

        BooleanIndexing.and(arr3, Conditions.greaterThanOEqual(5));  

        * We can also check if at least one value in the array meets our condition. This should return false.  

        BooleanIndexing.or(arr3, Conditions.isNan());


17. Now try swapping out backends to use GPUs. This is as easy as changing one line in your POM.xml. The current version of Jcuda requires you to specify your CUDA version (supporting versions 5.5, 6.0 and 6.5). For example, if you have CUDA v6.0 installed, then you need to define the artifactId as:

        <dependency>
          <groupId>org.nd4j</groupId>
          <artifactId>nd4j-jcublas-6.5</artifactId>
          <version>${nd4j.version}</version>
        </dependency>

You can replace the <artifactId> ... </artifactId>, depending on your preference:  

        nd4j-jcublas-$CUDA_VERSION (where CUDA_VERSION is one of 5.5, 6.0, or 6.5)

That’s it. Switching to GPUs means changing one line in your POM.xml file.

18. axpy: Compute y <- alpha * x + y (elementwise addition)  

        INDArray axpy = Nd4j.getBlasWrapper().axpy(2, arr, arr2);

19. ND4J Op Executioner: Accumulations, Transforms, and Scalar Operations  

        * Accumulation (add):  

        INDArray arr4 = Nd4j.linspace(1, 6, 6);
        Sum sum = new Sum(arr4);
        double sum2 = Nd4j.getExecutioner().execAndReturn(sum).currentResult().doubleValue();  

        * Transform: Square all values in the array.  

        Pow pow = new Pow(arr4, 2);  
        Nd4j.getExecutioner().exec(pow).z();  

        * Op executioner: Scalar Operation  

        arr4 = Nd4j.linspace(1, 6, 6);
        Nd4j.getExecutioner().exec(new ScalarAdd(arr4, 1));

##**Data Vectorization**

**Why do We Need to Vectorize Data?**

Throughout this course and while working with machine learning, we may need to work with many types of data (text, time series, audio, image, video, etc.). A key requirement is being able to take any type of data, and represent it as a vector.  

Before we get too far ahead of ourselves, let’s define what a vector is. We define a vector as:  

        For a positive integer n, a vector is an n-tuple, ordered (multi)set, or array of n numbers, called elements or scalars.  

We want to create a vector from a raw dataset via a process called vectorization. The number of elements in the vector is called the “order” (or “length”) of the vector.  

Here is an example of raw data from the canonical Iris dataset, which represents species of flowers:  

        5.1,3.5,1.4,0.2,Iris-setosa 4.9,3.0,1.4,0.2,Iris-setosa 4.7,3.2,1.3,0.2,Iris-setosa 4.6,3.1,1.5,0.2,Iris-setosa 5.0,3.6,1.4,0.2,Iris-setosa 5.4,3.9,1.7,0.4,Iris-setosa 4.6,3.4,1.4,0.3,Iris-setosa 5.0,3.4,1.5,0.2,Iris-setosa 4.4,2.9,1.4,0.2,Iris-setosa 4.9,3.1,1.5,0.1,Iris-setosa

Another example (raw text document):  

        Shall I compare thee to a summer's day?
        Thou art more lovely and more temperate:
        Rough winds do shake the darling buds of May,
        And summer's lease hath all too short a date.

Both these examples of raw data require some level of vectorization to be used for machine learning. We want our machine learning algorithm’s input data to look more like the serialized sparse vector format below:  

        -1 1:0.43 3:0.12 9284:0.2 

A very common question is, “Why do machine learning algorithms want the data represented (typically) as a (sparse) matrix?” To understand that, let’s make a quick detour into the basics of linear algebra. We are interested in solving systems of linear equations of the form  

        Ax = b  

where A is a matrix, or set of input row vectors, and b is the column vector of labels for each vector in the A matrix. 

Raw source data typically needs to be transformed before it can be modeled. The data can take on a number of manifestations, including:

* Raw text, such as a document in a text file
* A file containing a tweet string per line
* Binary time series data in a custom file format
* Pre-processed datasets with a mixture of numeric and string attributes
* Image files
* Audio files  

Depending on the condition and source of the raw data, attributes can be in numeric or string representation. In most cases, the values of attributes are numeric and continuous. These attributes either measure numbers’ real or integer value. 

Datasets have a mixture of raw attribute types and require pre-processing. To create vectors from this raw input data, we need to select the proper features of the data that we think most relevant to the model. Feature selection (or “feature engineering”) has long been held as a key to building a successful model. The number of features in our produced vector typically will not match up to the number of attributes in the source data. Many times the source data will be joined together with other datasets. In that case, a subset of attributes of the resultant denormalized view of the dataset will be used to derive the final set of features in our vector.

In standard vectorization, we create a fixed size n-length vector (where n minus 1 is our feature count, and the last slot is dedicated to the label value). We then set the values of each indexed cell according to some pre-determined heuristic to represent our data. The process of vectorization is based on selecting attributes and finding a dimension in a feature vector to assign the feature to. The process of taking n attributes and turning them into m features in our output vector can be done in a number of ways; this is referred to as “feature engineering”. Feature engineering techniques include:

* Taking the unchanged values directly from the attribute 
* Feature scaling, standardization, or normalizing an attribute to create a feature
* Binarization of features
* Dimensionality reduction
* Prewhitening

##**Getting to Know Canova**
Canova is a framework built from the ground up to assist in the creation of data pipelines for machine learning algorithms.

Canova has a baseline idea of a RecordReader. A record consists of a collection of Writables. 

Writables represent primitives such as floats, double, and Strings.

In this exercise, we are going to create record readers for the different types of data mentioned earlier.

Specifically: images, text, audio, and video.



##**Text Data**

You are probably familiar with bag of words by now. In this exercise, we will be creating a bag of words data set using [reuters](http://www.daviddlewis.com/resources/testcollections/reuters21578/).

Reuters is a standard text classification dataset. It is used in quite a few tasks to benchmark text classification algorithms.

In this exercise, we will be creating bag of words with our nd4j representation.

1. Create a new project in intellij with the maven archetype steps we went over earlier.
2. Include the following artifact:
     
        <dependency>
        <artifactId>canova-nd4j-nlp</artifactId>
        <groupId>org.nd4j</groupId>
        <version>${canova.version}</version>
        </dependency>

3. Now let's create a TfidfVectorizer. TFIDF if we remember stands for term frequency inverse document frequency. This is a weighting scheme used for words. Higher weights mean more importance in a document.

4. Type ctrl shift t in the ide. This will bring up a dialog for which you can search. Let's first start with a configuration.

5. Create a Configuration object. Resolve the import using the ide (alt + enter)

6. Create a TfidfVectorizer. Initialize it with an empty configuration.

7. Let's try it out with a basic run now.

8. Call fit as follows:
       
          RecordReader reader = new CollectionRecordReader(Writables.writables(Arrays.asList("Testing one.", "Testing 2.")));
          INDArray n = vectorizer.fitTransform(reader);

9. Let's break that down a bit: We created a collection record reader with the documents that we want to do tfidf on.
    
   This gave us a 2 by 3 document representing the bag of words.

10. Let's set this up with reuters now. You should find this in the data directory.

11. Use the ArchiveUtils to extract it to /tmp

12. Now we can use a FileInputSplit to load the data. Create a record reader and initialize it with a FileRecordReader.
    In the FileRecordReader pass the path to the directory:
    /tmp/20news-18828
    
    
    This will loop over all of the documents and tokenizer them and calculate tfidf scores for every document and word.
    

13. Now we need to set the following keys in the configuration:
           
           TextVectorizer.MIN_WORD_FREQUENCY
           
    We want to set a minimum word frequency otherwise the vocab will balloon in size. This also leads to less noisy data.


   Now let's call fit:
   
           INDArray tfidf = vectorizer.fitTransform(fileRecordReader);



14. Now inspect your tifdf matrix that got returned.

15. Feel free to play around with different word frequencies to understand the relationships involve.d

16. Now let's run this again, this time customizing the tokenizer.

17. The goal of this will be to see the difference between the default tokenizer (split on white space) vs what a machine learning based tokenizer that knows how to say: segment periods from other words and the results this could give us.
    
18. Set the new tokenizer as follows:

          conf.set(TextVectorizer.TOKENIZER, UimaTokenizerFactory.class.getName());

    Ensure you also recall each initialize function.
    
    Compare the shapes of the output to see what the delta was on the vocab.
    




##**Video Data (Time series)**
Reading video data into Canova is similar to the process of loading text data.  

1. Create a new Java class inside of src/main/java. Call this class VectorizeVideoData.  

2. Load video data from UCSD_Anomaly_Dataset.tar.gz  

3. Create an FileInputSplit from the path: 
      
              File baseDir = new File(System.getProperty("java.io.tmpdir"));
             File ucsdDir = new File(baseDir,"UCSD_Anomaly_Dataset.v1p2/UCSDped1/Train/");

4. Create a VideoRecordReader specifying the size.  Calling sequenceRecord() will give you back a sequence of records where each collection represents a frame of a video.

##**Image Data**

1. Create a new Java class inside of src/main/java. Call this class VectorizeImageData.

2. Create an ImageRecordReader with a height and width of 28 x 28.

3. Load image data from data/lfw.tgz  

            The layout of the data is similar to what we want for video. Each person/face is a directory.

4. Now load 2 images from the record reader (call .next() on 2 different records)

5. Using the RecordConverter, convert both of the images, and perform a convolution using Nd4j.getConvolution()




##**Audio Data**

Reading audio data into Canova is also similar to other data types.

1. Create a new Java class inside of src/main/java. Call this class VectorizeAudioData.  

2. Load audio data from mlsp_contest_dataset.zip.  

            The unzipped directory contains audio files in essential_data/src_wavs

3. Unzip the data. 

4. Create a WavFileRecordReader. Initialize it with a FileSplit.

##**Extra Credit**

Another way to vectorize text data is using Word2Vec. The Word2Vec method takes text in the form of an input corpus and outputs “word vectors.” The algorithm starts by building a vocabulary from the input training data and then builds the representations of the individual words. With Word2Vec, the output from training on the input corpus is the unique set of words in the corpus with a vector attached to each word. Each vector in this output contains the word’s context (or usage).  

1. Load your data.  

                ClassPathResource resource = new ClassPathResource("raw_sentences.txt");

2. Create a LineSentenceIterator with your loaded file.  

                SentenceIterator sentenceIterator = new LineSentenceIterator(resource.getFile());

3. Set a SentencePreProcessor for your SentenceIterator. Consider lowercasing your text and stripping punctuation here. This will be applied to all sentences in your dataset.  

                SentenceIterator sentenceIterator = new LineSentenceIterator(resource.getFile());
                sentenceIterator.setPreProcessor(new SentencePreProcessor() {
                    @Override
                    public String preProcess(String sentence) {
                        String loweredSentence = sentence.toLowerCase();
                        loweredSentence = StringCleaning.stripPunct(loweredSentence);
                        return loweredSentence;
                    }
                });

4. Create a UimaTokenizerFactory.  

                TokenizerFactory tokenizerFactory = new UimaTokenizerFactory();

5. Set a TokenPreProcessor for your TokenizerFactory. Here, you might want to consider stemming your tokens.    

                final EndingPreProcessor endingPreProcessor = new EndingPreProcessor();
                tokenizerFactory.setTokenPreProcessor(new TokenPreProcess() {
                    @Override
                    public String preProcess(String token) {
                        String stemmedToken = endingPreProcessor.preProcess(token);
                        stemmedToken = stemmedToken.replaceAll("\\d", "d");
                        return stemmedToken;
                    }
                });

6.  Build Word2Vec with your SentenceIterator and TokenizerFactory.  

                Word2Vec vec = new Word2Vec.Builder()
                .minWordFrequency(1)
                .iterations(4)
                .stopWords(StopWords.getStopWords())
                .learningRate(0.01)
                .minLearningRate(1e-4)
                .iterate(sentenceIterator)
                .tokenizerFactory(tokenizerFactory)
                .build();

7. Train your Word2Vec vectorizer.  

        vectorizer.fit();

