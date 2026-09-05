package com.federatedlearning;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * Loads breast cancer patient data from a hospital's CSV file.
 * Each row = one patient's tumor measurements + diagnosis.
 */
public class DataLoader {

    // Holds one patient's data: 10 tumor measurements + diagnosis label
    public static class Sample {
        public double[] features; // 10 mean tumor measurements
        public int label;         // 0 = benign, 1 = malignant

        public Sample(double[] features, int label) {
            this.features = features;
            this.label = label;
        }
    }

    /**
     * Reads a hospital's CSV file and returns all patient samples.
     * Each hospital (client) calls this with ITS OWN file only.
     */
    public static List<Sample> loadAll(String csvPath) throws IOException {
        List<Sample> samples = new ArrayList<>();

        try (BufferedReader br = new BufferedReader(new FileReader(csvPath))) {
            String line;
            boolean isHeader = true;

            while ((line = br.readLine()) != null) {
                if (isHeader) {
                    isHeader = false; // skip the header row
                    continue;
                }
                if (line.trim().isEmpty()) continue;

                String[] parts = line.split(",");

                double[] features = new double[10];
                for (int i = 0; i < 10; i++) {
                    features[i] = Double.parseDouble(parts[i]);
                }

                String diagnosis = parts[10].trim();
                int label = diagnosis.equals("malignant") ? 1 : 0;

                samples.add(new Sample(features, label));
            }
        }

        return samples;
    }
}