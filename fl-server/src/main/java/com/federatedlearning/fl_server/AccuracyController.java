package com.federatedlearning.fl_server;

import org.springframework.web.bind.annotation.*;

import java.util.*;

@RestController
public class AccuracyController {

    // round -> (clientName -> accuracy)
    private final Map<Integer, Map<String, Double>> history = new TreeMap<>();

    /**
     * Clients call this after evaluating their accuracy each round.
     */
    @PostMapping("/report-accuracy")
    public synchronized String reportAccuracy(@RequestBody AccuracyReport report) {
        history.computeIfAbsent(report.round, k -> new LinkedHashMap<>())
                .put(report.clientName, report.accuracy);
        return "Accuracy recorded.";
    }

    /**
     * Dashboard polls this to get all data needed to render charts.
     */
    @GetMapping("/dashboard-data")
    public Map<String, Object> getDashboardData() {
        Map<String, Object> result = new LinkedHashMap<>();
        result.put("history", history); // round -> clientName -> accuracy
        return result;
    }

    public static class AccuracyReport {
        public String clientName;
        public int round;
        public double accuracy;

        public AccuracyReport() {}
    }
}