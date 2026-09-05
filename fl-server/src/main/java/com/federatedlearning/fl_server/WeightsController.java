package com.federatedlearning.fl_server;

import org.springframework.web.bind.annotation.*;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

@RestController
public class WeightsController {

    private static final int EXPECTED_CLIENTS = 3;

    // Stores weights submitted by each client
    private final Map<String, double[]> submittedWeights = new HashMap<>();

    // Stores number of local records for each client
    private final Map<String, Integer> clientDataSizes = new HashMap<>();

    // Latest global model
    private double[] globalWeights = null;

    // Current federated learning round
    private int currentRound = 1;

    /**
     * Client submits its locally trained model.
     */
    @PostMapping("/submit-weights")
    public synchronized String submitWeights(
            @RequestBody WeightRequest request) {

        System.out.println();
        System.out.println("----------------------------------------");
        System.out.println("Received model from: " + request.clientName);
        System.out.println("Round: " + request.round);
        System.out.println("Local records: " + request.dataSize);

        // Check round
        if (request.round != currentRound) {

            return "ERROR: Expected round "
                    + currentRound
                    + " but received round "
                    + request.round;
        }

        // Check weights
        if (request.weights == null ||
                request.weights.length == 0) {

            return "ERROR: Empty model weights received.";
        }

        // Store client model
        submittedWeights.put(
                request.clientName,
                request.weights
        );

        clientDataSizes.put(
                request.clientName,
                request.dataSize
        );

        System.out.println(
                "Clients received: "
                        + submittedWeights.size()
                        + "/"
                        + EXPECTED_CLIENTS
        );

        // Wait until all clients submit
        if (submittedWeights.size() < EXPECTED_CLIENTS) {

            return "Received from "
                    + request.clientName
                    + ". Waiting for remaining clients.";
        }

        // All clients received
        System.out.println();
        System.out.println(
                "All 3 clients submitted their models."
        );

        performFedAvg();

        // Clear current round
        submittedWeights.clear();
        clientDataSizes.clear();

        currentRound++;

        System.out.println(
                "Global model created successfully."
        );

        System.out.println(
                "Server ready for round "
                        + currentRound
        );

        System.out.println("----------------------------------------");

        return "Received. FedAvg completed. Global model updated.";
    }

    /**
     * Client fetches the latest global model.
     */
    @GetMapping("/global-weights")
    public synchronized double[] getGlobalWeights() {

        if (globalWeights == null) {

            System.out.println(
                    "Global model is not available yet."
            );

            return new double[0];
        }

        System.out.println(
                "Global model requested by a client."
        );

        return globalWeights;
    }

    /**
     * Performs weighted Federated Averaging.
     *
     * Formula:
     *
     * Global Model =
     * (W1*N1 + W2*N2 + W3*N3) / (N1+N2+N3)
     */
    private void performFedAvg() {

        List<double[]> allWeights =
                new ArrayList<>(submittedWeights.values());

        List<Integer> allDataSizes =
                new ArrayList<>(clientDataSizes.values());

        int weightLength =
                allWeights.get(0).length;

        double[] global =
                new double[weightLength];

        int totalData = 0;

        // Calculate total number of records
        for (int size : allDataSizes) {
            totalData += size;
        }

        System.out.println(
                "Total training records: "
                        + totalData
        );

        // Weighted average
        for (int clientIndex = 0;
             clientIndex < allWeights.size();
             clientIndex++) {

            double[] clientWeights =
                    allWeights.get(clientIndex);

            int clientDataSize =
                    allDataSizes.get(clientIndex);

            double clientWeight =
                    (double) clientDataSize / totalData;

            System.out.println(
                    "Client contribution weight: "
                            + clientWeight
            );

            for (int i = 0;
                 i < weightLength;
                 i++) {

                global[i] +=
                        clientWeights[i]
                                * clientWeight;
            }
        }

        globalWeights = global;

        System.out.println(
                "Federated Averaging complete."
        );
    }

    /**
     * Returns server status.
     */
    @GetMapping("/status")
    public synchronized String getStatus() {

        return "Federated Learning Server is running. "
                + "Current round: "
                + currentRound
                + ", Clients received: "
                + submittedWeights.size()
                + "/"
                + EXPECTED_CLIENTS;
    }

    /**
     * Data received from a client.
     */
    public static class WeightRequest {

        public String clientName;

        public int dataSize;

        public int round;

        public double[] weights;

        public WeightRequest() {
        }
    }
}