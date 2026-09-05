package com.federatedlearning;

import java.util.ArrayList;
import java.util.List;

public class Main {

    public static void main(String[] args) throws Exception {

        System.out.println("========================================");
        System.out.println(
                "  FEDERATED LEARNING: BREAST CANCER DETECTION");
        System.out.println("========================================\n");

        List<DataLoader.Sample> hospital1Data =
                DataLoader.loadAll(
                        "src/main/resources/client1_cancer_hospital.csv");

        List<DataLoader.Sample> hospital2Data =
                DataLoader.loadAll(
                        "src/main/resources/client2_checkup_clinic.csv");

        List<DataLoader.Sample> hospital3Data =
                DataLoader.loadAll(
                        "src/main/resources/client3_community_hospital.csv");

        List<Client> clients = new ArrayList<>();

        clients.add(
                new Client(
                        "Cancer-Specialty-Hospital",
                        hospital1Data
                )
        );

        clients.add(
                new Client(
                        "General-Checkup-Clinic",
                        hospital2Data
                )
        );

        clients.add(
                new Client(
                        "Community-Hospital",
                        hospital3Data
                )
        );

        System.out.println();

        System.out.println(
                "Note: Each hospital has a DIFFERENT mix of cases (Non-IID data)"
        );

        System.out.println();

        System.out.println(
                "Starting federated training rounds over the network...\n"
        );

        System.out.println(
                "Open http://localhost:8080/dashboard.html in your browser to watch live!\n"
        );

        int numRounds = 8;
        int localEpochsPerRound = 3;

        for (int round = 1;
             round <= numRounds;
             round++) {

            System.out.println(
                    "--- Federated Learning Round "
                            + round
                            + " ---"
            );

            // Local training at each hospital
            for (Client client : clients) {
                client.trainLocally(localEpochsPerRound);
            }

            System.out.println();

            // Send local models to Spring Boot server
            for (Client client : clients) {
                client.sendWeightsToServer(round);
            }

            System.out.println();

            System.out.println(
                    "All client models sent to the server."
            );

            // Receive global model from server
            for (Client client : clients) {
                client.fetchGlobalWeightsFromServer();
            }

            double averageAccuracy = 0.0;

            System.out.println();

            // Evaluate updated global model
            for (Client client : clients) {

                double accuracy =
                        client.evaluateAccuracy();

                averageAccuracy += accuracy;

                System.out.printf(
                        "   %s accuracy: %.2f%%%n",
                        client.getClientName(),
                        accuracy * 100
                );

                // Report to server so the dashboard can display it
                client.reportAccuracyToServer(round, accuracy * 100);
            }

            averageAccuracy /= clients.size();

            System.out.printf(
                    "Round %d/%d -> Average global accuracy: %.2f%%%n%n",
                    round,
                    numRounds,
                    averageAccuracy * 100
            );
        }

        System.out.println(
                "========================================"
        );

        System.out.println(
                "  FEDERATED LEARNING COMPLETE"
        );

        System.out.println(
                "========================================"
        );

        System.out.println();

        System.out.println(
                "3 hospitals collaboratively trained a cancer-detection model."
        );

        System.out.println(
                "Patient records were kept locally at each hospital."
        );

        System.out.println(
                "Only model weights were sent to the central server."
        );

        System.out.println(
                "The server performed weighted Federated Averaging."
        );
    }
}