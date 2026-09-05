package com.federatedlearning;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.util.ArrayList;
import java.util.List;

import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;

import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

public class Client {

    private static final String SERVER_URL =
            "http://localhost:8080";

    private final String clientName;
    private final List<DataLoader.Sample> localData;

    private MultiLayerNetwork model;

    private double[] featureMean;
    private double[] featureStd;

    private final HttpClient httpClient;

    public Client(
            String clientName,
            List<DataLoader.Sample> localData) {

        this.clientName = clientName;
        this.localData = localData;

        System.out.println(
                clientName
                        + " has "
                        + localData.size()
                        + " local patient records (never shared)"
        );

        httpClient = HttpClient.newHttpClient();

        createModel();
        computeNormalizationStats();
    }

    private void createModel() {

        MultiLayerConfiguration configuration =
                new NeuralNetConfiguration.Builder()
                        .seed(123)
                        .updater(new Adam(0.001))
                        .list()

                        .layer(
                                new DenseLayer.Builder()
                                        .nIn(10)
                                        .nOut(20)
                                        .activation(Activation.RELU)
                                        .build()
                        )

                        .layer(
                                new DenseLayer.Builder()
                                        .nIn(20)
                                        .nOut(10)
                                        .activation(Activation.RELU)
                                        .build()
                        )

                        .layer(
                                new OutputLayer.Builder(
                                        LossFunctions.LossFunction.XENT)
                                        .nIn(10)
                                        .nOut(1)
                                        .activation(Activation.SIGMOID)
                                        .build()
                        )

                        .build();

        model = new MultiLayerNetwork(configuration);
        model.init();
    }

    private void computeNormalizationStats() {
        int numFeatures = 10;
        featureMean = new double[numFeatures];
        featureStd = new double[numFeatures];

        int n = localData.size();
        if (n == 0) return;

        for (DataLoader.Sample sample : localData) {
            for (int i = 0; i < numFeatures; i++) {
                featureMean[i] += sample.features[i];
            }
        }
        for (int i = 0; i < numFeatures; i++) {
            featureMean[i] /= n;
        }

        for (DataLoader.Sample sample : localData) {
            for (int i = 0; i < numFeatures; i++) {
                double diff = sample.features[i] - featureMean[i];
                featureStd[i] += diff * diff;
            }
        }
        for (int i = 0; i < numFeatures; i++) {
            featureStd[i] = Math.sqrt(featureStd[i] / n);
            if (featureStd[i] == 0) featureStd[i] = 1;
        }
    }

    private double[] normalize(double[] rawFeatures) {
        double[] normalized = new double[rawFeatures.length];
        for (int i = 0; i < rawFeatures.length; i++) {
            normalized[i] = (rawFeatures[i] - featureMean[i]) / featureStd[i];
        }
        return normalized;
    }

    public String getClientName() {
        return clientName;
    }

    public int getLocalDataSize() {
        return localData.size();
    }

    // ---------------------------------------------------------
    // LOCAL TRAINING
    // ---------------------------------------------------------

    public void trainLocally(int epochs) {

        System.out.println(
                "   " + clientName
                        + " is training locally..."
        );

        List<DataSet> dataSetList =
                new ArrayList<>();

        for (DataLoader.Sample sample : localData) {

            INDArray input =
                    Nd4j.create(
                            new double[][]{
                                    normalize(sample.features)
                            });

            INDArray output =
                    Nd4j.create(
                            new double[][]{
                                    {sample.label}
                            });

            dataSetList.add(
                    new DataSet(input, output)
            );
        }

        if (dataSetList.isEmpty()) {

            System.out.println(
                    "   No local data available for "
                            + clientName
            );

            return;
        }

        for (int epoch = 0;
             epoch < epochs;
             epoch++) {

            for (DataSet dataSet :
                    dataSetList) {

                model.fit(dataSet);
            }
        }

        System.out.println(
                "   " + clientName
                        + " completed local training."
        );
    }

    // ---------------------------------------------------------
    // LOCAL MODEL WEIGHTS
    // ---------------------------------------------------------

    public INDArray getModelWeights() {
        return model.params().dup();
    }

    // ---------------------------------------------------------
    // SEND WEIGHTS TO SPRING BOOT SERVER
    // ---------------------------------------------------------

    public void sendWeightsToServer(int round)
            throws Exception {

        System.out.println(
                "   " + clientName
                        + " sending model weights to server..."
        );

        double[] weights =
                model.params().toDoubleVector();

        StringBuilder json =
                new StringBuilder();

        json.append("{");

        json.append("\"clientName\":\"")
                .append(escapeJson(clientName))
                .append("\",");

        json.append("\"dataSize\":")
                .append(localData.size())
                .append(",");

        json.append("\"round\":")
                .append(round)
                .append(",");

        json.append("\"weights\":[");

        for (int i = 0; i < weights.length; i++) {

            if (i > 0) {
                json.append(",");
            }

            json.append(weights[i]);
        }

        json.append("]");

        json.append("}");

        HttpRequest request =
                HttpRequest.newBuilder()
                        .uri(
                                URI.create(
                                        SERVER_URL
                                                + "/submit-weights"
                                )
                        )
                        .header(
                                "Content-Type",
                                "application/json"
                        )
                        .POST(
                                HttpRequest.BodyPublishers
                                        .ofString(
                                                json.toString()
                                        )
                        )
                        .build();

        HttpResponse<String> response =
                httpClient.send(
                        request,
                        HttpResponse.BodyHandlers.ofString()
                );

        System.out.println(
                "   Server response: "
                        + response.body()
        );

        if (response.statusCode() != 200) {

            throw new RuntimeException(
                    "Server returned HTTP "
                            + response.statusCode()
            );
        }
    }

    // ---------------------------------------------------------
    // FETCH GLOBAL MODEL FROM SERVER
    // ---------------------------------------------------------

    public void fetchGlobalWeightsFromServer()
            throws Exception {

        System.out.println(
                "   " + clientName
                        + " fetching global model from server..."
        );

        HttpRequest request =
                HttpRequest.newBuilder()
                        .uri(
                                URI.create(
                                        SERVER_URL
                                                + "/global-weights"
                                )
                        )
                        .GET()
                        .build();

        HttpResponse<String> response =
                httpClient.send(
                        request,
                        HttpResponse.BodyHandlers.ofString()
                );

        if (response.statusCode() != 200) {

            throw new RuntimeException(
                    "Server returned HTTP "
                            + response.statusCode()
            );
        }

        double[] globalWeights =
                parseDoubleArray(response.body());

        if (globalWeights.length == 0) {

            throw new RuntimeException(
                    "Global model is not available yet."
            );
        }

        INDArray globalModel =
                Nd4j.create(globalWeights);

        model.setParams(globalModel);

        System.out.println(
                "   " + clientName
                        + " received global model."
        );
    }

    // ---------------------------------------------------------
    // COMPATIBILITY METHOD
    // ---------------------------------------------------------

    public void receiveGlobalWeights(
            INDArray globalWeights) {

        model.setParams(
                globalWeights.dup()
        );

        System.out.println(
                "   " + clientName
                        + " received global model."
        );
    }

    // ---------------------------------------------------------
    // EVALUATION
    // ---------------------------------------------------------

    public double evaluateAccuracy() {

        if (localData.isEmpty()) {
            return 0.0;
        }

        int correct = 0;

        for (DataLoader.Sample sample :
                localData) {

            INDArray input =
                    Nd4j.create(
                            new double[][]{
                                    normalize(sample.features)
                            });

            INDArray prediction =
                    model.output(input);

            double predictedValue =
                    prediction.getDouble(0);

            int predictedClass =
                    predictedValue >= 0.5
                            ? 1
                            : 0;

            int actualClass =
                    sample.label >= 0.5
                            ? 1
                            : 0;

            if (predictedClass ==
                    actualClass) {

                correct++;
            }
        }

        return (double) correct /
                localData.size();
    }

    public MultiLayerNetwork getModel() {
        return model;
    }

    // ---------------------------------------------------------
    // REPORT ACCURACY TO SERVER (for dashboard)
    // ---------------------------------------------------------

    public void reportAccuracyToServer(int round, double accuracy) throws Exception {

        String json = "{"
                + "\"clientName\":\"" + escapeJson(clientName) + "\","
                + "\"round\":" + round + ","
                + "\"accuracy\":" + accuracy
                + "}";

        HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create(SERVER_URL + "/report-accuracy"))
                .header("Content-Type", "application/json")
                .POST(HttpRequest.BodyPublishers.ofString(json))
                .build();

        httpClient.send(request, HttpResponse.BodyHandlers.ofString());
    }

    // ---------------------------------------------------------
    // SIMPLE JSON PARSER
    // ---------------------------------------------------------

    private double[] parseDoubleArray(
            String json) {

        json = json.trim();

        if (json.equals("[]")) {
            return new double[0];
        }

        if (!json.startsWith("[")
                || !json.endsWith("]")) {

            throw new RuntimeException(
                    "Invalid global weights received: "
                            + json
            );
        }

        String content =
                json.substring(
                        1,
                        json.length() - 1
                ).trim();

        if (content.isEmpty()) {
            return new double[0];
        }

        String[] values =
                content.split(",");

        double[] result =
                new double[values.length];

        for (int i = 0;
             i < values.length;
             i++) {

            result[i] =
                    Double.parseDouble(
                            values[i].trim()
                    );
        }

        return result;
    }

    private String escapeJson(
            String text) {

        return text
                .replace("\\", "\\\\")
                .replace("\"", "\\\"");
    }
}