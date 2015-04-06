package org.deeplearning4j;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.accum.Sum;
import org.nd4j.linalg.api.ops.impl.scalar.ScalarAdd;
import org.nd4j.linalg.api.ops.impl.transforms.Pow;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.BooleanIndexing;
import org.nd4j.linalg.indexing.conditions.Conditions;
import org.nd4j.linalg.indexing.functions.Value;

import java.awt.image.DataBuffer;
import java.util.Arrays;

/**
 * <pre>
 *     Nd4j Example for SSA Class.
 *     Day 1. sections.
 * </pre>
 *
 * @author naimtenor
 * Created by ubuntu on 15. 4. 6.
 */
class Nd4jExamples {
    public static void main(String[] args) {

//        Nd4j.ENFORCE_NUMERICAL_STABILITY = true;
//        Nd4j.dtype = org.nd4j.linalg.api.buffer.DataBuffer.DOUBLE;

        INDArray a111 = Nd4j.ones(4);
        INDArray a112 = Nd4j.zeros(4);

        System.out.println(Nd4j.concat(0, a111, a112));
        System.out.println(Nd4j.cumsum(a111));
        /* Create a row vector with the specified number of columns. */
        INDArray arr = Nd4j.create(4);  // Nd4j.zeros(4) equal
        System.out.println("First arr : " + arr);

        /* Create a row vector with the specified number of columns (all values set to equal 1). */
        INDArray arr2 = Nd4j.ones(4);
        System.out.println("First arr2 : " + arr2);

        /* Create a row vector with 10 columns, ranging from 1 to 10. */
        INDArray arr3 = Nd4j.linspace(1, 10, 10);
        System.out.println("First arr3 : " + arr3);

        /* Add tow arrays. */
        // similar to "arr += 1"
        arr.add(arr2);
        System.out.println("arr add arr2 : " + arr);
        /* Add tow arrays in-place. Notice the difference int the method name. */
        // similar to "arr2 = arr1 + 1"
        arr.addi(arr2);
        System.out.println("arr add arr2 in-place : " + arr);

        /* Transpose a matrix. */
        INDArray arrT = arr.transpose();
        System.out.println("arr original : " + arr);
        System.out.println("arr transposed : \n" + arrT);

        /* Compute row (1) column (0) sums. */
        Nd4j.sum(arr2, 1);
        System.out.println("arr2 sum 1 : " + arr2);
        Nd4j.sum(arr2, 0);
        System.out.println("arr2 sum 0 : " + arr2);

        /* Check array shape. */
        System.out.println("current arr2 shape : " + Arrays.toString(arr2.shape()) + "\n");

        /* Assign the value 5 to each element of an array (just like Numby's "fill" method). */
        arr.assign(5);
        System.out.println("arr assign 5 : " + arr);

        /* Reshape the array. */
        arr2 = arr2.reshape(2, 2);
        System.out.println("arr2 reshape : \n" + arr2);
        System.out.println("current arr2 shape : " + Arrays.toString(arr2.shape()) + "\n");

        /* Sort the array. Also try sorting "and" returning sorted indices. */
        arr2 = Nd4j.sort(arr2, 0, true);
        System.out.println("sorted arr2 indices : \n" + Arrays.toString(Nd4j.sortWithIndices(arr2, 0, true)));

        /* Compute basic statistical properties (mean and standard deviation). */
        Nd4j.mean(arr);
        Nd4j.std(arr);

        /* Find min and max values. */
        Nd4j.max(arr3);
        Nd4j.min(arr3);

        /* Boolean indexing : Where a given condition holds true, apply a function to an NDArray. */
        // In this example, replace any values below 5 with 5
        BooleanIndexing.applyWhere(arr3, Conditions.lessThan(5), new Value(5));

        // In this example, replace any NaN values with 0
        BooleanIndexing.applyWhere(arr3, Conditions.isNan(), new Value(0));

        // Here, we can check if we successfully replaced every value less than 5. This should return true.
        BooleanIndexing.and(arr3, Conditions.greaterThanOEqual(5));

        // We can also check if at least one value in the array meets our condition.
        BooleanIndexing.or(arr3, Conditions.isNan());

        /* axpy : Compute y <- alpha * x + y (elementwise addition) */
        // This means "y = ax + b", => blah = ax + y
        INDArray axpy = Nd4j.getBlasWrapper().axpy(2, arr, arr2);

        /* ND4J Op Executioner : Accumulations, Transforms, and Scalar Operations */
        // Accumulation (add) :
        INDArray arr4 = Nd4j.linspace(1, 6, 6);
        Sum sum = new Sum(arr4);
        double sum2 = Nd4j.getExecutioner().execAndReturn(sum).currentResult().doubleValue();
        System.out.println(arr4);

        // Transform : Square all values in the array.
        Pow pow = new Pow(arr4, 2);
        Nd4j.getExecutioner().exec(pow).z();
        System.out.println(arr4);

        // Op executioner : Scalar Operation
        arr4 = Nd4j.linspace(1, 6, 6);
        Nd4j.getExecutioner().execAndReturn(new ScalarAdd(arr4, 1));
        System.out.println(arr4);
    }
}
