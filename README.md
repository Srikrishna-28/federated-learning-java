# Federated Learning for Breast Cancer Detection

A working simulation of **Federated Learning** — a privacy-preserving machine learning technique where multiple parties (hospitals, in this case) collaboratively train a shared model **without ever sharing their raw data**. Only model weights are exchanged, never patient records.

This is the same core technique used by Google (Gboard next-word prediction) and Apple (Siri, on-device personalization) to train models without centralizing user data.

## What This Project Does

Three simulated hospitals, each with a different patient population, jointly train a breast cancer classifier:

| Hospital | Patients | Data Mix |
|---|---|---|
| Cancer-Specialty-Hospital | 150 | 80% malignant cases |
| General-Checkup-Clinic | 200 | 85% benign cases |
| Community-Hospital | 219 | Balanced mix |

Each hospital trains a neural network on **only its own patient data**, then sends just the model's weights to a central server. The server performs **weighted Federated Averaging** (weighting each hospital's contribution by how much data it has) to produce an improved global model, which is sent back to every hospital. This repeats over multiple rounds, and accuracy is tracked live on a web dashboard.

## Architecture

```
┌─────────────────────┐         HTTP (REST API)        ┌──────────────────────┐
│   fl-server          │◄───────────────────────────────►│   fl-simulation       │
│   (Spring Boot)       │                                 │   (Java + DL4J)       │
│                        │                                 │                        │
│  • Receives weights    │                                 │  • 3 simulated        │
│    from each hospital  │                                 │    hospital clients    │
│  • Performs weighted   │                                 │  • Local model         │
│    Federated Averaging │                                 │    training (DL4J)     │
│  • Serves live         │                                 │  • Feature             │
│    dashboard data       │                                 │    normalization       │
└─────────────────────┘                                 └──────────────────────┘
```

**Tech stack:** Java 17, DeepLearning4J (DL4J), Spring Boot, REST APIs, HTML/JS + Chart.js dashboard.

## Key Engineering Details

- **Real dataset:** Wisconsin Breast Cancer dataset (10 tumor measurements per patient), not a toy dataset.
- **Non-IID data:** Each hospital's data distribution is deliberately different (like real hospitals), which surfaces a genuine, well-known federated learning challenge — a globally averaged model can under-perform on a client whose data distribution differs sharply from the rest.
- **Feature normalization:** Implemented per-client z-score normalization after discovering that raw, unscaled tumor measurements caused the network to collapse into always predicting the majority class.
- **Weighted Federated Averaging:** Hospitals with more patient records contribute proportionally more to the global model (data-size-weighted average, not a simple mean).
- **Live dashboard:** A browser dashboard (`/dashboard.html`) polls the server every 2 seconds and plots each hospital's accuracy per round using Chart.js.

## Project Structure

```
federated-learning-java/
├── fl-server/              # Spring Boot REST API server
│   └── src/main/java/com/federatedlearning/fl_server/
│       ├── WeightsController.java     # Receives weights, performs FedAvg
│       └── AccuracyController.java    # Feeds the live dashboard
├── fl-simulation/          # Simulated hospital clients
│   └── src/main/java/com/federatedlearning/
│       ├── Client.java     # Local training, normalization, HTTP communication
│       ├── DataLoader.java # Loads each hospital's CSV
│       └── Main.java       # Orchestrates training rounds
└── README.md
```

## How to Run

**1. Start the server:**
```
cd fl-server
mvn spring-boot:run
```

**2. Open the live dashboard in a browser:**
```
http://localhost:8080/dashboard.html
```

**3. In a separate terminal, start the simulation:**
```
cd fl-simulation
mvn exec:java -Dexec.mainClass="com.federatedlearning.Main"
```

Watch the dashboard update live as each hospital trains locally and the global model improves over 8 rounds.

## What I Learned

Building this surfaced two real, non-obvious ML engineering problems:

1. **Feature scaling matters more than expected** — without normalizing the tumor measurements (which range from ~0.05 to ~2500 across features), the network degenerated into a trivial "always predict majority class" solution, which looked like reasonable accuracy but wasn't learning anything.
2. **Federated Averaging isn't automatically fair across clients** — when one hospital's data distribution differs significantly from the others, the globally averaged model can perform worse for that hospital than a naive majority-class baseline would. This is an active research area (approaches like FedProx and personalized federated learning exist specifically to address it).

## Disclaimer

This is an educational simulation running on `localhost`, not a production or clinically validated system. The breast cancer dataset is the well-known public Wisconsin Diagnostic Breast Cancer dataset, used here purely to demonstrate the federated learning technique.
