package com.federatedlearning;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class App {
    public static void main(String[] args) {
        System.out.println("Testing DL4J / ND4J setup...");
        
        // Create a simple array to test ND4J is working
        INDArray array = Nd4j.create(new double[]{1.0, 2.0, 3.0, 4.0}, new int[]{2, 2});
        
        System.out.println("Array created successfully:");
        System.out.println(array);
        System.out.println("DL4J/ND4J setup is working correctly!");
    }
}