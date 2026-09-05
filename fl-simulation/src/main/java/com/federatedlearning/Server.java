package com.federatedlearning;

import java.util.List;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class Server {

    private INDArray globalWeights;

    // FedAvg aggregation
    public void aggregate(
            List<INDArray> clientWeightsList,
            List<Integer> clientDataSizes) {

        if (clientWeightsList.isEmpty()) {
            throw new IllegalArgumentException(
                    "No client weights provided");
        }

        if (clientWeightsList.size() != clientDataSizes.size()) {
            throw new IllegalArgumentException(
                    "Weights and data sizes must match");
        }

        int totalData = 0;

        for (int size : clientDataSizes) {
            totalData += size;
        }

        if (totalData == 0) {
            throw new IllegalArgumentException(
                    "Total client data cannot be zero");
        }

        // Start with zero weights
        globalWeights =
                Nd4j.zeros(clientWeightsList.get(0).shape());

        // Weighted FedAvg
        for (int i = 0; i < clientWeightsList.size(); i++) {

            INDArray clientWeights =
                    clientWeightsList.get(i);

            double weight =
                    (double) clientDataSizes.get(i) / totalData;

            globalWeights.addi(
                    clientWeights.mul(weight)
            );
        }

        System.out.println(
                "   Server completed FedAvg aggregation.");
    }

    // Return global model
    public INDArray getGlobalWeights() {

        if (globalWeights == null) {
            throw new IllegalStateException(
                    "Global model has not been created yet.");
        }

        return globalWeights.dup();
    }
}