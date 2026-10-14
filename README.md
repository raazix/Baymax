# LineGuard (Baymax)

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python: 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Next.js: 16](https://img.shields.io/badge/Next.js-16-black?logo=next.js&logoColor=white)](https://nextjs.org/)
[![React: 19](https://img.shields.io/badge/React-19-61DAFB?logo=react&logoColor=black)](https://react.dev/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.0%2B-3178C6?logo=typescript&logoColor=white)](https://www.typescriptlang.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-CUDA_Enabled-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Ultralytics YOLO11](https://img.shields.io/badge/YOLO11n-Ultralytics-00FFFF?logo=ultralytics&logoColor=black)](https://docs.ultralytics.com/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-17-4169E1?logo=postgresql&logoColor=white)](https://www.postgresql.org/)
[![SQLAlchemy: 2.0](https://img.shields.io/badge/SQLAlchemy-2.0-D71F00?logo=sqlalchemy&logoColor=white)](https://www.sqlalchemy.org/)
[![Tests: 125+ Passing](https://img.shields.io/badge/Tests-125%2B_Passing-success)](tests/)

**LineGuard** is an evidence-linked quality inspection, root-cause analysis (RCA), and predictive quality platform developed for industrial manufacturing environments (Singularity 2026, Track 3: Automotive Quality Inspection). Designed around safety-critical components such as automotive brake discs and precision cast parts, LineGuard bridges the operational divide between raw deep-learning detections and accountable shop-floor decision making.

Every defect detection is subjected to deterministic engineering severity rules, linked to telemetry-driven root-cause hypotheses via TreeSHAP, bounded by statistical defect forecasting via PLSR and Monte Carlo bootstrap, and anchored to an immutable, cryptographically hash-chained audit ledger.

---

## Table of Contents

1. [Overview & Problem Statement](#1-overview--problem-statement)
2. [Key Capabilities](#2-key-capabilities)
3. [High-Level System Architecture](#3-high-level-system-architecture)
4. [End-to-End Inspection & Decision Lifecycle](#4-end-to-end-inspection--decision-lifecycle)
5. [AI & Machine Learning Architecture](#5-ai--machine-learning-architecture)
6. [Computer Vision Pipeline](#6-computer-vision-pipeline)
7. [Analytics, Explainability & Predictive Risk](#7-analytics-explainability--predictive-risk)
8. [Evidence, Traceability & Cryptographic Audit](#8-evidence-traceability--cryptographic-audit)
9. [Database Architecture & Schema](#9-database-architecture--schema)
10. [Frontend Architecture & User Interface](#10-frontend-architecture--user-interface)
11. [Backend & Worker Architecture](#11-backend--worker-architecture)
12. [Deployment Architecture](#12-deployment-architecture)
13. [Repository Structure](#13-repository-structure)
14. [Technology Stack](#14-technology-stack)
15. [Local Development & Setup Guide](#15-local-development--setup-guide)
16. [Environment Variables](#16-environment-variables)
17. [Interactive Demo Walkthrough](#17-interactive-demo-walkthrough)
18. [Testing & Quality Assurance](#18-testing--quality-assurance)
19. [Implementation Status](#19-implementation-status)
20. [Known Limitations & Engineering Truths](#20-known-limitations--engineering-truths)
21. [Roadmap](#21-roadmap)
22. [Security & Authentication](#22-security--authentication)
23. [Contributing](#23-contributing)
24. [License](#24-license)
25. [Acknowledgements & References](#25-acknowledgements--references)
26. [Project Status & Engineering Disclaimer](#26-project-status--engineering-disclaimer)

---

## 1. Overview & Problem Statement

### The Problem
Traditional computer vision in manufacturing is plagued by three systemic failures:
1. **Uncalibrated Model Confidence:** Raw neural network output probabilities (e.g., softmax scores) are frequently mistaken for physical severity. A high-confidence cosmetic scratch can trigger a false plant shutdown, while a low-confidence hair-line crack in a critical structural zone goes unflagged.
2. **The "Black Box" Attribution Gap:** When a vision model flags a defect, quality engineers on the shop floor lack the causal context needed to correct the machine process. Without real-time telemetry correlation, defects recur in subsequent lots.
3. **Absence of Verifiable Proof:** Most quality stations generate ephemeral bounding boxes without cryptographically binding the raw image bytes, model artifact versions, process telemetry, and engineer sign-offs into an auditable evidence chain.

### The LineGuard Approach
LineGuard replaces opaque end-to-end automation with an **evidence-first, human-in-the-loop decision station**:
- **Deterministic Severity Triage:** Model confidence is strictly decoupled from quality severity. Severity is evaluated deterministically based on defect class, measured physical/geometric size (mm), part zone (`rim`, `hub`, `friction face`), and multiplicity.
- **Explainable Process Telemetry:** Inspection findings are correlated with live or historical sensor parameters (`temperature_c`, `pressure_bar`, `vibration_mm_s`, `machine_speed_rpm`) using XGBoost and exact **TreeSHAP** attribution values.
- **Predictive Quality Forecasting:** Partial Least Squares Regression (**PLSR**) and **Monte Carlo** residual bootstrapping estimate future-lot defect rates and tolerance breach probabilities before scrap is produced.
- **Cryptographic Traceability:** Every inspected frame, model run, operator decision, and 20-part verification is locked into an append-only, SHA-256 hash-chained audit trail.

---

## 2. Key Capabilities

### Computer Vision & Anomaly Detection
- **Multi-Model Inference Router:** Dynamically routes frames to fine-tuned **YOLO11n** object detectors, unsupervised **PatchCore** feature-distance anomaly detectors, or a dedicated **ResNet-18 corrosion classifier**.
- **Calibrated Image Quality Gate:** Pre-inference validation using Laplacian blur variance and specular overexposure rejection to prevent garbage-in/garbage-out degradation.
- **SAHI (Slicing Aided Hyper Inference):** Sliced tile inference (128–2048 px tiles with adjustable overlap) for small defect detection on high-resolution steel surfaces.
- **Spatial Geometry & Metrology:** Extracts polar coordinates $(r, \theta)$, defect bounding diameters, and outer/inner radial zones (`rim`, `hub`, `vane_root`).

### Analytics & Predictive Risk
- **Root Cause Analysis (RCA):** Identifies process failure modes (`thermal_process_drift`, `pressure_instability`, `tooling_vibration`, `speed_drift`) with signed TreeSHAP margins.
- **Lot Defect Forecasting:** PLSR 3-lot autoregressive regression benchmarked against PCR and persistence baselines.
- **Monte Carlo Uncertainty Engine:** 10,000 seeded empirical residual resamples producing 95% predictive intervals and tolerance breach probabilities.
- **Fleet-Wide Risk Ranking:** Aggregates telemetry across multiple manufacturing stations (M-01 through M-04) to rank equipment by risk.
- **Interactive Process Lab ("What-If"):** Interactive parameter sandbox enabling engineers to simulate telemetry deviations and observe live TreeSHAP and Monte Carlo response curves without writing to production storage.

### Action, Governance & Audit
- **Deterministic SOP Engine:** Maps defect classes and root-cause hypotheses to standard operating procedures and containment actions (part hold vs. full lot quarantine).
- **Human-in-the-Loop Gate:** Decisions (`approve`, `reject`, `escalate`) require recorded engineer identification. Direct PLC overrides are forbidden by design.
- **20-Part Evidence Verification:** Verifies corrective action efficacy by requiring at least 20 consecutive passed inspections from the same machine post-approval.
- **Immutable Audit Hash-Chain:** Transactional projection where every event is cryptographically linked to its predecessor via SHA-256 digests.

### Multimodal & Extended Interaction
- **Grounded LLM Assistant:** NVIDIA Nemotron-backed assistant answering natural language inquiries grounded strictly in deterministic JSON inspection evidence.
- **Critical Spoken Alerts:** Automated ElevenLabs voice synthesis reading stored evidence upon critical quality holds.
- **OpenCV.js WebGL Tracking:** In-browser 2D silhouette tracking and anomaly heatmap projection.
- **Remote Mobile Gateway:** Secure QR pairing over Cloudflare HTTPS tunnels allowing real-time phone camera scanning.

---

## 3. High-Level System Architecture

```mermaid
flowchart TD
    subgraph ClientLayer["Client & Edge Layer"]
        Browser["Desktop Operator Browser<br/>(Next.js 16 Dashboard)"]
        Mobile["Operator Mobile / Tablet<br/>(PWA / Camera Scanning)"]
        OpenCV["OpenCV.js + WebGL<br/>(2D Silhouette Tracking)"]
        ThreeJS["Three.js Engine<br/>(3D Production Line & Rotor)"]
    end

    subgraph GatewayLayer["Networking & Gateway"]
        ReverseProxy["Next.js Route Handler / Reverse Proxy<br/>(/api/* Forwarding & Stream Throttling)"]
        CFTunnel["Cloudflare Zero-Trust Tunnel<br/>(Temporary Phone Demo Pairing)"]
    end

    subgraph AppServer["FastAPI Application Server (Port 8000)"]
        Middleware["Security & Context Middleware<br/>(Request-ID, Bearer Auth Token)"]
        
        subgraph APIRoutes["REST API Endpoints"]
            F_Routes["/api/frames & /api/camera"]
            I_Routes["/api/inspections & /api/replay"]
            M_Routes["/api/proxy & /api/patchcore"]
            A_Routes["/api/analytics & /api/history"]
            G_Routes["/api/assistant & /api/models"]
        end

        subgraph CoreDomain["Core Domain Services"]
            QualityGate["Calibrated Quality Gate<br/>(Laplacian Blur / Specularity)"]
            UploadOrch["Inspection Orchestrator<br/>(upload_inspection.py)"]
            SeverityRules["Severity Rules Engine v3<br/>(Class, Size, Zone, Multiplicity)"]
            SOPEngine["Deterministic SOP Engine<br/>(Containment & Corrective Steps)"]
            PartRef["Part Reference Registry<br/>(Brake Disc SHA-256 Identification)"]
        end
    end

    subgraph VisionSubsystem["Computer Vision & Deep Learning"]
        YOLO["YOLO11n Proxy Detector<br/>(NEU Steel Surface / SAHI)"]
        PatchCore["PatchCore Anomaly Detector<br/>(ResNet-18 Coreset Memory Banks)"]
        Rust["Dual-Head Corrosion Classifier<br/>(ResNet-18 Color-Gated)"]
        Metrology["Polar Metrology Engine<br/>(Radial Zone Mapping & Scale)"]
    end

    subgraph AnalyticsSubsystem["Analytics & Statistical Engine"]
        Worker["Durable Async Analytics Worker<br/>(analytics.worker)"]
        XGB["XGBoost RCA Classifier<br/>(Process Drift Hypotheses)"]
        SHAP["Exact TreeSHAP Explainer<br/>(Feature Margin Contributions)"]
        PLSR["PLSR / PCR Forecaster<br/>(Next-Lot Defect Fraction)"]
        MonteCarlo["Monte Carlo Simulator<br/>(10k Residual Resamples)"]
        FleetRisk["Fleet Risk Engine<br/>(Multi-Machine Ranking)"]
    end

    subgraph DataLayer["Persistence & Storage Layer"]
        Repo["SQLAlchemy 2.0 Repository"]
        DB[(PostgreSQL 17 / Neon / SQLite)]
        FramesStore[("Raw Frame Blobs & Hashes")]
        AuditChain[("Append-Only Audit Ledger<br/>(SHA-256 Hash Chain)")]
    end

    subgraph ExternalServices["External Multimodal Integrations"]
        NVIDIA["NVIDIA NIM API<br/>(Nemotron-3-Ultra / Llama-3.2-Vision)"]
        ElevenLabs["ElevenLabs Audio API<br/>(Critical TTS / Voice STT)"]
        Supermemory["Supermemory Cloud<br/>(Approved Action Precedent Store)"]
    end

    Browser --> ReverseProxy
    Mobile --> CFTunnel --> ReverseProxy
    Browser -.-> OpenCV
    Browser -.-> ThreeJS
    ReverseProxy --> Middleware --> APIRoutes

    APIRoutes --> UploadOrch
    UploadOrch --> QualityGate
    UploadOrch --> VisionSubsystem
    VisionSubsystem --> Metrology --> SeverityRules
    UploadOrch --> AnalyticsSubsystem
    AnalyticsSubsystem --> SOPEngine

    APIRoutes --> Repo
    Worker --> Repo
    Repo --> DB
    DB --- FramesStore
    DB --- AuditChain

    APIRoutes -.-> ExternalServices
```

---

## 4. End-to-End Inspection & Decision Lifecycle

### Lifecycle Data Flow
```mermaid
flowchart LR
    A["1. Image Source<br/>(Upload / Camera)"] --> B["2. Quality Gate<br/>(Blur / Exposure)"]
    B -- "Failed" --> B1["Disposition: Recapture<br/>(Save Frame Evidence)"]
    B -- "Passed" --> C["3. Vision Inference<br/>(YOLO / PatchCore / Rust)"]
    C --> D["4. Spatial Geometry<br/>(Polar Coordinates & Scale)"]
    D --> E["5. Deterministic Severity<br/>(v3 Engineering Rules)"]
    E --> F["6. Telemetry Correlation<br/>(Historical Lots / Presets)"]
    F --> G["7. RCA & Predictive Risk<br/>(TreeSHAP & PLSR / Monte Carlo)"]
    G --> H["8. SOP Generation<br/>(Lot / Part Containment)"]
    H --> I["9. Engineer Decision<br/>(Approve / Reject / Escalate)"]
    I -- "Approved" --> J["10. 20-Part Verification<br/>(Sequential Clean Parts)"]
    J --> K["11. Cryptographic Audit<br/>(Immutable Hash Chain)"]
```

### Inspection Decision State Machine
```mermaid
stateDiagram-v2
    [*] --> Ingested: Upload / Camera Scan
    
    Ingested --> QualityRejected: Blur / Specularity Breach
    QualityRejected --> [*]: Frame Stored (Disposition: Recapture)
    
    Ingested --> QualityPassed: Quality Gate Verified
    
    state QualityPassed {
        [*] --> VisionProcessing: Dispatch Frame
        VisionProcessing --> SpatialMerge: YOLO + PatchCore + Rust
        SpatialMerge --> SeverityCalculation: Apply Rules v3
        SeverityCalculation --> ProcessTelemetryJoin: Attach Machine Context
        ProcessTelemetryJoin --> AnalyticsInference: TreeSHAP + PLSR + MC
        AnalyticsInference --> SOPRecommendation: Map Containment Rules
        SOPRecommendation --> [*]
    }

    QualityPassed --> PendingDecision: Action Required (Defects Detected)
    QualityPassed --> AutoCleared: Clean Component (No Defects)
    AutoCleared --> [*]: Disposition: Pass

    state PendingDecision {
        [*] --> UnderReview
        UnderReview --> Approved: Engineer Sign-off
        UnderReview --> Rejected: Engineer Discard
        UnderReview --> Escalated: Supervisor Review Required
    }

    Rejected --> [*]: Audit Event Logged
    Escalated --> [*]: Audit Event Logged

    Approved --> VerificationPending: Require >= 20 Clean Parts
    
    state VerificationPending {
        [*] --> Sampling: Ingest Consecutive Parts
        Sampling --> VerificationPassed: 20/20 Parts Defect-Free
        Sampling --> VerificationFailed: Defect Encountered in Sample
    }

    VerificationPassed --> VerificationRecorded: Upper Bound Rate Calculated
    VerificationFailed --> PendingDecision: Re-trigger Lot Quarantine
    VerificationRecorded --> [*]: Production Line Resumed
```

---

## 5. AI & Machine Learning Architecture

LineGuard employs a heterogeneous collection of computer vision, classical machine learning, and statistical models. Every model operates within declared architectural boundaries:

```mermaid
flowchart TD
    subgraph VisionModels["Vision & Anomaly Models"]
        YOLO_M["YOLO11n (NEU Proxy)<br/>Input: 200x200 or Sliced Tiles<br/>Output: 6 Bounding Box Classes<br/>Weights: Best rebalanced checkpoint"]
        PatchCore_M["PatchCore (ResNet-18)<br/>Input: 224x224 RGB Features<br/>Output: 28x28 Anomaly Distance Grid<br/>Memory: 1024x384 Coreset Vectors"]
        Rust_M["ResNet-18 Rust Classifier<br/>Input: Cropped RGB Image<br/>Output: Corrosion / Severe Probabilities<br/>Gate: Color Saturation Check"]
    end

    subgraph AnalyticsModels["Process Analytics & Forecasting"]
        XGB_M["XGBoost Classifier<br/>Input: 4 Process Features<br/>Output: 4 Failure Modes + Nominal<br/>Format: Universal Binary JSON (.ubj)"]
        SHAP_M["TreeSHAP Explainer<br/>Input: Booster Trees + Instance<br/>Output: Raw Class Margin Attributions<br/>Constraint: Non-causal Ranked Correlation"]
        PLSR_M["PLSR Forecaster (n=2)<br/>Input: 3 Consecutive Lots Telemetry<br/>Output: Next-Lot Defect Fraction<br/>Benchmark: Principal Component Reg. (PCR)"]
        MC_M["Monte Carlo Residual Bootstrap<br/>Input: PLSR Prediction + Held-out Residuals<br/>Output: 10k Simulation Array, 95% Interval<br/>Metric: Tolerance Breach Probability"]
    end

    subgraph MultimodalServices["Multimodal Intelligence Layer"]
        LLM_M["NVIDIA Nemotron-3-Ultra / Llama-3.2<br/>Input: Deterministic Briefing JSON Packet<br/>Output: Grounded Natural Language Answers<br/>Guardrail: Read-Only, No Decision Power"]
        Voice_M["ElevenLabs Voice Engine<br/>Input: Canonical Defect Summary<br/>Output: 44.1 kHz Alert Audio (TTS)<br/>Condition: Severity >= High"]
    end

    VisionModels --> MergedEvidence["Spatial & Evidence Aggregator"]
    AnalyticsModels --> MergedEvidence
    MergedEvidence --> DecisionEngine["Deterministic Severity & SOP Engine"]
    MergedEvidence -.-> MultimodalServices
```

### Model Provenance & Truth Matrix

| Model Asset | Artifact Path | Training Dataset | Input Dimension | Output Metric | Scope & Limitations |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **YOLO11n (NEU)** | `models/yolo/active.json` $\to$ `best.pt` | NEU-DET (1,800 steel surface images) | $200\times 200$ (or sliced tiles) | 6 surface defect classes, bounding boxes | **Proxy model.** Evaluated on steel sheets; does not establish automotive brake-disc accuracy. |
| **PatchCore (Casting)** | `models/patchcore/casting_proxy.pt` | 363 normal casting impeller images | $224\times 224$ (ResNet-18 L2/L3) | Continuous anomaly distance & $28\times 28$ grid | **Proxy model.** Anomaly distance is not a calibrated probability; grid is not a segmentation mask. |
| **PatchCore (MPDD)** | `models/patchcore/mpdd/*/casting_proxy.pt` | MPDD (6 categories: plate, connector, etc.) | $224\times 224$ (ResNet-18 L2/L3) | Feature distance vs. category threshold | **Proxy model.** Metal plate model serves as default component demo proxy. |
| **PatchCore (MVTec)** | `models/patchcore/mvtec_metal_nut/*.pt` | MVTec AD (`metal_nut` subset) | $224\times 224$ (ResNet-18 L2/L3) | AUROC 0.9892, Recall 87.1% | Evaluated strictly on industrial nuts. Rotor 3D viewer hidden when active. |
| **Corrosion Classifier** | `models/rust/model.pt` | Roboflow Rust v1 (de-duplicated) | $224\times 224$ RGB | Corrosion & severe corrosion scores | Color-gated: skips monochrome steel to eliminate false positives. |
| **XGBoost RCA** | `models/xgboost/model.ubj` | 500 synthetic batches $\times$ 12 parts | 4 telemetry float values | 5 failure mode class probabilities | **Synthetic scenario classifier.** Uncalibrated probabilities; does not prove physical causality. |
| **TreeSHAP Explainer** | Dynamic native booster computation | XGBoost tree ensembles | Telemetry vector + class index | Signed feature margin contributions | Explains model scoring margins, not physical root causes. |
| **PLSR Forecaster** | `models/plsr/forecast.joblib` | Ordered synthetic lot histories | 3 rolling lot telemetry records | Predicted next-lot defect fraction | **Synthetic history.** Forecasts defect fraction, not calibrated machine failure probability. |
| **Monte Carlo Bootstrap** | Dynamic statistical engine | 360 held-out calibration residuals | Prediction + residual distribution | 95% predictive interval & breach risk | Assumes exchangeable residual distribution conditional on synthetic inputs. |

---

## 6. Computer Vision Pipeline

```mermaid
flowchart TD
    RawInput["Raw Frame Source<br/>(JPEG / PNG Upload or Live WebRTC Capture)"] --> IngestionCheck["Image Ingestion Engine<br/>(image_ingestion.py)"]
    
    IngestionCheck --> ByteLimit{"File Size <= 8 MiB &<br/>Dimension Valid?"}
    ByteLimit -- "No" --> HTTP413["Reject HTTP 413 / 422"]
    ByteLimit -- "Yes" --> DecodeRGB["OpenCV Decode (BGR -> RGB)"]
    
    DecodeRGB --> QualityCheck["Quality Gate (quality_gate.py)"]
    
    subgraph QualityGateExecution["Quality Gate Evaluation"]
        Laplacian["Laplacian Variance<br/>(Blur Detection)"]
        Specularity["Specular Saturation<br/>(Pixel Value >= 250 Fraction)"]
        ProfileCheck["Profile Baseline Matching<br/>(camera / neu_proxy / casting_proxy)"]
    end
    
    QualityCheck --> QualityGateExecution
    QualityGateExecution --> QualityPassed{"Quality Gate<br/>Passed?"}
    
    QualityPassed -- "No" --> QualityFailStore["Save Frame with Rejection Reasons<br/>(Return Disposition: Recapture)"]
    QualityPassed -- "Yes" --> ModelRouter["Vision Model Dispatcher"]
    
    subgraph ParallelVision["Vision Execution"]
        subgraph YOLO_Path["YOLO11n Detection"]
            YOLO_Select{"Mode?"}
            YOLO_Select -- "Full Image" --> YOLO_Full["Standard Forward Pass"]
            YOLO_Select -- "SAHI Sliced" --> YOLO_Slice["Greedy NMM Tile Slicing<br/>(128-2048px, Overlap 0-0.5)"]
            YOLO_Full --> YOLO_Out["Boxes + Class Confidences"]
            YOLO_Slice --> YOLO_Out
        end
        
        subgraph PatchCore_Path["PatchCore Anomaly Detection"]
            Resize224["Resize 224x224 & Normalize"] --> ResNetFeat["Extract ResNet-18 Layer 2 & 3"]
            ResNetFeat --> CoresetDist["Euclidean Nearest-Neighbor vs Coreset Bank"]
            CoresetDist --> AnomalyGrid["28x28 Distance Map & Max Score"]
        end
        
        subgraph Rust_Path["Corrosion Analysis"]
            ColorGate{"Color Gate<br/>(Sat > 18 & Std > 8)?"}
            ColorGate -- "Monochrome" --> RustSkip["Skip (NEU Steel Protection)"]
            ColorGate -- "Color" --> RustInfer["ResNet-18 Dual Head Inference"]
            RustInfer --> RustOut["Corrosion Confidence Scores"]
        end
    end
    
    ModelRouter --> ParallelVision
    ParallelVision --> SpatialMerge["Spatial Fusion & Geometry (part_geometry.py)"]
    
    subgraph SpatialAnalysis["Spatial Metrology & Merging"]
        Contour["Extract Outer / Inner Radial Contours"]
        Polar["Compute Polar Coordinates (r, theta)"]
        ZoneAssign["Classify Part Zone (rim, hub, friction face)"]
        BoxMerge["Merge Bounding Box with Anomaly Clusters"]
    end
    
    SpatialMerge --> SpatialAnalysis
    SpatialAnalysis --> SeverityEngine["Deterministic Severity Rules v3 (severity.py)"]
    
    subgraph SeverityEvaluation["Severity Triage Matrix"]
        BaseClass["Assign Class Base Severity"]
        SizeCheck["Escalate by Dimension (>= 5mm, >= 15mm)"]
        ZoneCheck["Escalate by Critical Zone (rim / hub)"]
        MultCheck["Escalate by Multiplicity (>= 3 Defects)"]
        CapRule["Enforce Safety Caps<br/>(Only Cracks reach Critical; Unknown capped at High)"]
    end
    
    SeverityEngine --> SeverityEvaluation
    SeverityEvaluation --> FinalEvidence["Packaged Vision Evidence Record"]
```

---

## 7. Analytics, Explainability & Predictive Risk

```mermaid
flowchart TD
    subgraph InputContext["Process Context Resolution"]
        LiveTelem["Live Inspection Telemetry<br/>(T, P, V, RPM)"]
        HistLots["Historical Sensor Database<br/>(46,080 Readings across M-01 to M-04)"]
        PresetSim["Synthetic Process Presets<br/>(Nominal / Thermal Drift / Tooling Wear)"]
    end

    InputContext --> ContextBuilder["Process Context Builder (lot_history.py)"]
    ContextBuilder --> FeatureVec["Engineered Telemetry Feature Vector"]

    subgraph RCA_Engine["Root Cause Analysis (analytics/xgboost_rca.py)"]
        XGB_Model["XGBoost Multi-Class Booster<br/>(models/xgboost/model.ubj)"]
        TreeSHAP["Exact Native TreeSHAP Engine<br/>(analytics/treeshap_explainer.py)"]
        
        FeatureVec --> XGB_Model
        XGB_Model --> ProbDist["Class Probabilities<br/>(Explicitly Uncalibrated)"]
        XGB_Model & FeatureVec --> TreeSHAP
        TreeSHAP --> SHAP_Values["Signed Margin Attributions per Feature<br/>(dT, dP, dV, dRPM)"]
    end

    subgraph Forecast_Engine["Predictive Quality (analytics/plsr_forecast.py)"]
        RollingLots["3-Lot Rolling History Window<br/>(Lot[t-2], Lot[t-1], Lot[t])"]
        PLSR_Model["Partial Least Squares Regressor<br/>(models/plsr/forecast.joblib)"]
        PCR_Model["Principal Component Regressor<br/>(Comparative Benchmark)"]
        
        RollingLots --> PLSR_Model
        RollingLots --> PCR_Model
        PLSR_Model --> DefectPred["Expected Next-Lot Defect Fraction"]
        PCR_Model --> BenchmarkPred["Benchmark Defect Fraction"]
    end

    subgraph Uncertainty_Engine["Uncertainty & Risk (analytics/monte_carlo.py)"]
        HeldoutResid["Held-out Calibration Residuals<br/>(360 Empirical Validation Errors)"]
        MC_Bootstrap["Seeded Residual Bootstrap<br/>(10,000 Iterations)"]
        
        DefectPred & HeldoutResid --> MC_Bootstrap
        MC_Bootstrap --> Interval95["95% Simulated Predictive Interval"]
        MC_Bootstrap --> BreachProb["Tolerance Breach Probability<br/>(P(Defect Rate > 0.50))"]
    end

    RCA_Engine --> AnalyticsService["Analytics Aggregator Service (analytics/service.py)"]
    Forecast_Engine --> AnalyticsService
    Uncertainty_Engine --> AnalyticsService
    AnalyticsService --> SOP["SOP Recommendation Engine (actions/sop_engine.py)"]
```

---

## 8. Evidence, Traceability & Cryptographic Audit

Every event in LineGuard is recorded in an immutable, append-only hash chain. Inconsistent modifications or injected records immediately break cryptographic integrity:

```mermaid
flowchart LR
    subgraph EvidenceInputs["Raw Evidence Generation"]
        ImgBytes["Raw Image Bytes"] --> ImgHash["Image SHA-256 Digest"]
        ModelArtifacts["Loaded Model Weights"] --> ModelHash["Artifact SHA-256 Digest"]
        InspectionData["Inspection Payload"] --> PayloadHash["Inspection Snapshot"]
    end

    subgraph HashChain["Append-Only Transactional Hash Ledger"]
        Event0["Event 0: inspection_created<br/>Seq: 0 | Prev: 00000000..."]
        Event1["Event 1: model_inference_completed<br/>Seq: 1 | Prev: Hash(Event 0)"]
        Event2["Event 2: engineer_decision<br/>Seq: 2 | Prev: Hash(Event 1)"]
        Event3["Event 3: verification_recorded<br/>Seq: 3 | Prev: Hash(Event 2)"]
        
        Event0 --> Event1 --> Event2 --> Event3
    end

    subgraph AuditVerification["Integrity Verification (/api/inspections/{id}/audit)"]
        Verifier["Chain Verifier Algorithm"]
        Status["Integrity Status: Verified / Compromised"]
        
        Event3 --> Verifier --> Status
    end

    EvidenceInputs --> HashChain
```

### Mathematical Hash-Chain Specification
For sequence index $i$:
$$\text{EventHash}_0 = \text{SHA256}(\text{"0"}^{64} + \text{JSON}(\text{Event}_0) + \text{"0"})$$
$$\text{EventHash}_i = \text{SHA256}(\text{EventHash}_{i-1} + \text{JSON}(\text{Event}_i) + \text{str}(i))$$

---

## 9. Database Architecture & Schema

LineGuard utilizes SQLAlchemy 2.0 with unified support for SQLite (local standalone development) and PostgreSQL 17 (production and multi-process deployments).

```mermaid
erDiagram
    FRAMES ||--o{ MODEL_RUNS : "evaluates"
    INSPECTIONS ||--|| PARTS : "inspects"
    INSPECTIONS ||--|| PROCESS_TELEMETRY : "records"
    INSPECTIONS ||--|| RISK_AND_RCA : "produces"
    INSPECTIONS ||--|| CORRECTIVE_ACTIONS : "triggers"
    INSPECTIONS ||--|| INSPECTION_REVISIONS : "tracks"
    INSPECTIONS ||--o{ AUDIT_EVENTS : "records"
    INSPECTIONS ||--o| ANALYTICS_JOBS : "schedules"
    SENSOR_DATASETS ||--o{ HISTORICAL_SENSOR_READINGS : "contains"

    FRAMES {
        string id PK "UUID4"
        string created_at "ISO-8601 UTC"
        string sha256 "Indexed SHA-256"
        string media_type "image/jpeg, image/png"
        blob content "Raw image binary"
        json quality "Quality gate results"
    }

    MODEL_RUNS {
        string id PK "UUID4"
        string frame_id FK "References frames.id"
        string created_at "ISO-8601 UTC"
        json payload "Raw model output & metrics"
    }

    INSPECTIONS {
        string id PK "UUID4"
        string created_at "ISO-8601 UTC"
        json payload "Complete canonical snapshot"
    }

    PARTS {
        string id PK "Physical Part ID"
        string lot_id "Manufacturing Lot ID"
        string machine_id "Manufacturing Machine ID"
        string source "synthetic_replay / uploaded_image"
    }

    PROCESS_TELEMETRY {
        string inspection_id PK, FK "References inspections.id"
        string part_id FK "References parts.id"
        string observed_at "ISO-8601 UTC"
        json values "T, P, V, RPM readings"
    }

    RISK_AND_RCA {
        string inspection_id PK, FK "References inspections.id"
        json payload "TreeSHAP, PLSR, Monte Carlo"
    }

    CORRECTIVE_ACTIONS {
        string inspection_id PK, FK "References inspections.id"
        string status "pending, approved, rejected, escalated"
        json payload "SOP steps & engineer note"
    }

    INSPECTION_REVISIONS {
        string inspection_id PK, FK "References inspections.id"
        int version "Optimistic lock version counter"
    }

    AUDIT_EVENTS {
        string id PK "UUID4"
        string inspection_id FK "References inspections.id"
        int sequence "Monotonic sequence number"
        json event "Event payload details"
        string previous_hash "Previous event SHA-256"
        string event_hash "Current event SHA-256"
    }

    ANALYTICS_JOBS {
        string id PK "UUID4"
        string inspection_id FK "Unique reference"
        string status "queued, running, completed, failed"
        string created_at "ISO-8601 UTC"
        string error "Nullable error trace"
    }

    SENSOR_DATASETS {
        string sha256 PK "Dataset content hash"
        string filename "Source CSV filename"
        string source_label "Provenance description"
        datetime imported_at "Timestamp with timezone"
        int rows_read "Total parsed rows"
        int rows_inserted "Inserted rows"
        int rows_duplicate "Ignored duplicate rows"
    }

    HISTORICAL_SENSOR_READINGS {
        string id PK "Deterministic SHA-256 ID"
        string dataset_sha256 FK "References sensor_datasets"
        string machine_id "Machine identifier"
        string sensor_name "Sensor parameter name"
        datetime observed_at "Timestamp with timezone"
        float value "Measured numeric value"
        string unit "Unit of measure"
        string lot_id "Optional lot reference"
    }
```

---

## 10. Frontend Architecture & User Interface

The frontend is built on **Next.js 16** and **React 19**, designed around industrial ergonomics and calm, evidence-first Material 3 principles ([DESIGN.md](DESIGN.md)).

```mermaid
flowchart TD
    subgraph DashboardApp["Next.js App Router (apps/dashboard/app)"]
        Layout["Root Layout (layout.tsx)"]
        Providers["Workspace Providers (WorkspaceProviders.tsx)"]
        MainPage["Main Workspace Console (page.tsx)"]
        
        Layout --> Providers --> MainPage
        
        subgraph ViewPanels["Workspace Views & Tabs"]
            InspectTab["Inspection Workspace<br/>(UploadedPanel.tsx & InspectionImage.tsx)"]
            RiskTab["Production Risk & Fleet Ranking<br/>(LineRisk.tsx & Charts.tsx)"]
            ProcessLabTab["Process Lab What-If Sandbox<br/>(ProcessLab.tsx)"]
            HistoryTab["Historical Sensor Catalog<br/>(HistorySensors.tsx)"]
            CameraTab["Camera & Scanning Station<br/>(CameraScanner.tsx & CameraInference.tsx)"]
            TrainingTab["Few-Shot PatchCore Training<br/>(TrainPatchCore.tsx)"]
            AuditTab["Evidence & Hash-Chain Trace<br/>(PipelineTrace.tsx)"]
        end

        subgraph InteractiveModules["Interactive & 3D Visualizers"]
            ThreeScene["Three.js 3D Manufacturing Lifecycle<br/>(ProductionLine.tsx)"]
            RotorScene["Three.js Procedural Rotor Viewer<br/>(RotorViewer.tsx)"]
            ARModule["OpenCV.js 2D Tracking & WebGL<br/>(TrackedAR.tsx & arWebGL.ts)"]
            AssistantModule["Multimodal AI Assistant<br/>(InspectionAssistant.tsx)"]
            PhoneModule["QR Phone Connect Gateway<br/>(PhoneConnect.tsx)"]
        end
        
        MainPage --> ViewPanels
        MainPage --> InteractiveModules
    end

    subgraph ClientDataLayer["Client Network Layer"]
        FetchClient["Typed API Client (api<T>)"]
        ProxyHandler["Next.js Reverse Proxy Route<br/>(app/api/[...path]/route.ts)"]
        
        ViewPanels --> FetchClient
        InteractiveModules --> FetchClient
        FetchClient --> ProxyHandler
    end

    ProxyHandler --> FastAPIBackend["FastAPI Backend (127.0.0.1:8000)"]
```

---

## 11. Backend & Worker Architecture

The backend consists of a synchronous FastAPI web server and a decoupled, durable background worker running against shared database persistence:

```mermaid
flowchart TD
    subgraph WebProcess["Uvicorn / FastAPI API Process (apps/api/main.py)"]
        RequestGate["Request Ingestion & Auth Middleware"]
        Router["FastAPI Route Handlers"]
        
        subgraph DomainServices["Service & Business Logic"]
            UploadSvc["Upload Orchestrator (upload_inspection.py)"]
            ReplaySvc["Synthetic Replay Generator (replay.py)"]
            BriefingSvc["Deterministic Briefing Engine (briefing.py)"]
            VisionAdapters["Vision Model Adapters (adapters.py)"]
        end
        
        RequestGate --> Router --> DomainServices
    end

    subgraph AsyncWorkerProcess["Durable Analytics Worker (analytics/worker.py)"]
        JobPoller["Claim Job Poller (repo.claim_job)"]
        JobExecutor["Analytics Executor (analytics.service.analyze)"]
        CrashRecovery["Crash Recovery (--recover flag)"]
        
        JobPoller --> JobExecutor
        CrashRecovery -.-> JobPoller
    end

    subgraph PersistenceLayer["Shared Storage Layer"]
        SQLRepo["SQLAlchemy Repository (database/repository.py)"]
        JobTable[("analytics_jobs Queue Table")]
        InspectionTable[("inspections Table")]
        
        DomainServices --> SQLRepo
        JobExecutor --> SQLRepo
        SQLRepo --> JobTable
        SQLRepo --> InspectionTable
    end
```

---

## 12. Deployment Architecture

```mermaid
flowchart TD
    subgraph ClientTier["Client Tier"]
        DesktopUser["Shop Floor Desktop / QC Engineer<br/>Chrome / Edge (HTTP / WebGL)"]
        MobileUser["Mobile Inspector / Tablet<br/>Safari / Chrome (HTTPS / WebRTC)"]
    end

    subgraph TunnelTier["Edge & Access Gateway (Optional Demo)"]
        Cloudflared["Cloudflare Named / Temporary Tunnel<br/>(HTTPS Termination & Port 3002 Forwarding)"]
    end

    subgraph HostServer["Host Workstation / Local Factory Edge (RTX 4050 Laptop / Linux Server)"]
        subgraph NodeTier["Node.js Runtime"]
            NextServer["Next.js Dashboard Server<br/>Port 3000 (Dev) / Port 3001/3002 (Demo)"]
        end

        subgraph PythonTier["Python Runtime (.venv)"]
            FastAPIServer["FastAPI / Uvicorn Server<br/>Port 8000 (4-8 Worker Threads)"]
            AnalyticsWorker["Standalone Analytics Worker<br/>python -m analytics.worker"]
        end

        subgraph ContainerTier["Docker Compose Infrastructure"]
            PostgresContainer["PostgreSQL 17 Database<br/>Port 5432 (Volume: lineguard_db)"]
        end
    end

    subgraph CloudAPIs["External SaaS APIs (Optional)"]
        NVIDIACache["NVIDIA NIM API (LLM)"]
        ElevenLabsAPI["ElevenLabs API (TTS/STT)"]
        SupermemoryAPI["Supermemory Cloud (Action Notes)"]
    end

    DesktopUser --> NextServer
    MobileUser --> Cloudflared --> NextServer
    NextServer --> FastAPIServer
    FastAPIServer --> PostgresContainer
    AnalyticsWorker --> PostgresContainer
    FastAPIServer -.-> CloudAPIs
```

---

## 13. Repository Structure

```
Baymax/
├── actions/                  # Corrective action recommendations and SOP execution
│   └── sop_engine.py         # Deterministic SOP matrix, quarantine rules, containment logic
├── analytics/                # Statistical, predictive, and root-cause analysis engines
│   ├── assistant.py          # NVIDIA Nemotron LLM & ElevenLabs audio integrations
│   ├── briefing.py           # Grounded deterministic JSON evidence packet builder
│   ├── calibration.py        # Uncertainty calibration helpers
│   ├── fleet_risk.py         # Multi-machine risk ranking engine
│   ├── historical_sensors.py # CSV sensor history parser and time-series aggregations
│   ├── lot_history.py        # Machine lot context and telemetry drift builder
│   ├── monte_carlo.py        # Seeded 10,000 residual-bootstrap uncertainty simulation
│   ├── plsr_forecast.py      # Partial Least Squares Regression defect fraction forecaster
│   ├── pcr_forecast.py       # Principal Component Regression benchmark forecaster
│   ├── service.py            # Unified analytics service facade
│   ├── supermemory.py        # Supermemory approved-action retrieval client
│   ├── treeshap_explainer.py # Exact TreeSHAP feature contribution calculation
│   ├── visual_assistant.py   # NVIDIA multimodal vision description client
│   ├── worker.py             # Durable background analytics worker daemon
│   └── xgboost_rca.py        # XGBoost multi-class process failure classifier
├── apps/                     # Application entry points
│   ├── api/                  # FastAPI backend server
│   │   └── main.py           # Primary REST API routes, middleware, and exception handlers
│   └── dashboard/            # Next.js 16 + React 19 web interface
│       ├── app/              # App router pages, workspace panels, and CSS stylesheets
│       │   ├── page.tsx      # Central dashboard console orchestrator
│       │   ├── UploadedPanel.tsx   # Inspection card and defect overlay
│       │   ├── LineRisk.tsx        # Production risk view and fleet ranking
│       │   ├── ProcessLab.tsx      # What-If parameter sandbox
│       │   ├── ProductionLine.tsx  # Three.js 3D manufacturing lifecycle simulation
│       │   ├── RotorViewer.tsx     # Three.js procedural 3D rotor model
│       │   ├── TrackedAR.tsx       # OpenCV.js 2D silhouette tracking and heatmap
│       │   ├── InspectionAssistant.tsx # Multimodal LLM chat & audio controls
│       │   ├── HistorySensors.tsx  # Historical telemetry catalog and charts
│       │   ├── TrainPatchCore.tsx  # Browser few-shot model training interface
│       │   └── PipelineTrace.tsx   # Evidence packet and audit hash-chain viewer
│       └── public/           # Static vendor assets (OpenCV.js 4.13.0, icons)
├── core/                     # Shared domain logic, schemas, and configurations
│   ├── clock.py              # UTC timestamp utilities
│   ├── config.py             # Environment configuration and database URL resolution
│   ├── part_reference.py     # Deterministic SHA-256 reference part identity resolver
│   ├── replay.py             # Deterministic synthetic replay fixture generator
│   ├── schemas.py            # Strict Pydantic v2 domain schemas and request models
│   └── upload_inspection.py  # Image upload orchestrator, geometry join, and triage
├── database/                 # Persistence layer and ORM models
│   ├── models.py             # SQLAlchemy 2.0 database tables and relationships
│   └── repository.py         # Data access object (CRUD, hash-chaining, atomic transactions)
├── demo_images/              # Hand-picked test images for demonstration workflows
├── docs/                     # 31 technical specification documents and model audits
│   ├── ASSISTANT.md          # LLM and audio provider integration architecture
│   ├── BACKEND.md            # Backend contracts, worker specs, and data guarantees
│   ├── MOBILE_UI.md          # Mobile layout, QR pairing, and animation guide
│   ├── MPDD.md               # MPDD component training benchmarks and recall analysis
│   ├── PATCHCORE.md          # PatchCore ResNet-18 mathematical and coreset documentation
│   ├── QUALITY_GATE.md       # Quality gate validation and profile calibrations
│   ├── SEVERITY.md           # Severity rules v3 engineering justification
│   ├── TRACK3.md             # Singularity Track 3 requirements audit and gap analysis
│   └── ...                   # Additional subsystem documents
├── models/                   # Trained weights, active registries, and metadata
│   ├── patchcore/            # Active PatchCore checkpoints (casting, MPDD, MVTec)
│   ├── plsr/                 # Trained PLSR forecasting model artifacts
│   ├── quality/              # Calibrated quality gate baseline thresholds
│   ├── reference_parts/      # Reference part identity hashes
│   ├── rust/                 # Trained ResNet-18 corrosion model weights
│   ├── xgboost/              # Trained XGBoost binary model (.ubj) and feature metadata
│   └── yolo/                 # Active YOLO11n checkpoint registry (active.json)
├── scripts/                  # 42 automation, training, calibration, and test scripts
│   ├── calibrate_patchcore.py # PatchCore threshold calibration CLI
│   ├── check_gpu.py          # CUDA and PyTorch GPU availability validator
│   ├── finetune_yolo.py      # YOLO11n fine-tuning pipeline
│   ├── generate_sensor_history.py # 46,080-row synthetic sensor history generator
│   ├── start_phone_demo.ps1  # Automated Cloudflare HTTPS tunnel and pairing launcher
│   ├── train_patchcore.py    # Standalone PatchCore coreset training pipeline
│   ├── train_xgboost.py      # XGBoost process failure mode training script
│   └── train_yolo.py         # Ultralytics YOLO training CLI
├── tests/                    # Comprehensive automated test suite (125+ tests)
├── docker-compose.yml        # PostgreSQL 17 container specification
├── requirements.txt          # Production backend dependencies
└── requirements-ml.txt       # Machine learning dependencies (PyTorch, Ultralytics, SAHI)
```

---

## 14. Technology Stack

| Layer | Technology | Version | Architectural Responsibility |
| :--- | :--- | :--- | :--- |
| **Frontend Framework** | [Next.js](https://nextjs.org/) | `^16.0.0` | App router, server-side layouts, reverse proxy |
| **UI Library** | [React](https://react.dev/) | `^19.0.0` | Declarative UI, hooks, concurrent rendering |
| **Type System** | [TypeScript](https://www.typescriptlang.org/) | `^5.0.0` | End-to-end interface and payload type safety |
| **Animation & UX** | [Motion](https://motion.dev/) | `^12.0.0` | Fluid micro-interactions, spring physics transitions |
| **3D Rendering** | [Three.js](https://threejs.org/) | `^0.174.0` | Procedural manufacturing lifecycle & 3D rotor scene |
| **In-Browser Vision** | [OpenCV.js](https://opencv.org/) | `4.13.0` | WebAssembly contour extraction & Lucas-Kanade optical flow |
| **Backend Framework**| [FastAPI](https://fastapi.tiangolo.com/) | `>=0.115,<1` | Asynchronous REST API, OpenAPI schema generation |
| **ASGI Server** | [Uvicorn](https://www.uvicorn.org/) | `>=0.30,<1` | Asynchronous worker process execution |
| **Validation Layer** | [Pydantic](https://docs.pydantic.dev/) | `v2` | Strict schema validation (`allow_inf_nan=False`, `extra='forbid'`) |
| **Object Detection** | [Ultralytics YOLO11](https://docs.ultralytics.com/) | `>=8.3,<8.4` | Anchor-free surface defect bounding box detection |
| **Tiled Inference** | [SAHI](https://github.com/obss/sahi) | `0.12.8` | Slicing Aided Hyper Inference for small defect detection |
| **Anomaly Detection**| [PyTorch](https://pytorch.org/) / torchvision | `2.6.0+cu126` | Compact ResNet-18 feature extraction & coreset indexing |
| **Tabular ML** | [XGBoost](https://xgboost.readthedocs.io/) | `>=3.0,<4` | Multi-class gradient boosted decision trees for RCA |
| **Explainability** | [TreeSHAP](https://github.com/shap/shap) | `>=0.46,<1` | Exact local feature attribution on tree ensembles |
| **Forecasting** | [scikit-learn](https://scikit-learn.org/) | `>=1.6,<2` | Partial Least Squares Regression (PLSR) & PCR |
| **ORM & DB Engine** | [SQLAlchemy](https://www.sqlalchemy.org/) | `>=2.0,<3` | Data mapping, atomic transactions, schema generation |
| **PostgreSQL Driver**| [psycopg3](https://www.psycopg.org/) | `>=3.2,<4` | High-performance binary PostgreSQL driver |
| **External LLM** | [NVIDIA NIM](https://build.nvidia.com/) | API | Nemotron-3-Ultra-550B & Llama-3.2-11B-Vision |
| **Voice Engine** | [ElevenLabs](https://elevenlabs.io/) | API | Low-latency Flash v2.5 TTS & Scribe v2 STT |
| **Precedent Store** | [Supermemory](https://supermemory.ai/) | API | Vectorized search over historical approved actions |

---

## 15. Local Development & Setup Guide

### Prerequisites
- **Operating System:** Windows 10/11, Ubuntu 22.04+, or macOS.
- **Python:** Python 3.11, 3.12, or 3.13.
- **Node.js:** Node.js 18.18+ or 20+.
- **Database (Optional):** Docker Desktop for PostgreSQL 17 (SQLite is used automatically if PostgreSQL is omitted).
- **GPU (Optional):** NVIDIA GPU with CUDA 12.4+ for hardware-accelerated PyTorch/YOLO inference.

---

### Step 1: Clone the Repository
```powershell
git clone https://github.com/raazix/Baymax.git
cd Baymax
```

---

### Step 2: Set Up Python Virtual Environment
```powershell
# Create virtual environment
python -m venv .venv

# Activate on Windows PowerShell:
.venv\Scripts\Activate.ps1

# Activate on Linux / macOS:
# source .venv/bin/activate

# Upgrade pip
python -m pip install --upgrade pip

# Install core backend dependencies:
python -m pip install -r requirements.txt

# (Optional) Install ML and CUDA training dependencies:
python -m pip install -r requirements-ml.txt
```

---

### Step 3: Configure Environment
Copy the example environment file:
```powershell
cp .env.example .env
```
*Note: SQLite is configured by default (`data/lineguard.db`). To use PostgreSQL, start the container and set `DATABASE_URL` in `.env`.*

```powershell
# Optional: Launch PostgreSQL 17 with Docker Compose
docker compose up -d
```

---

### Step 4: Install Dashboard Dependencies
In a separate terminal:
```powershell
cd apps/dashboard
npm install
cd ../..
```

---

### Step 5: Run the Backend & Analytics Worker
```powershell
# Terminal 1: Launch FastAPI Backend (Port 8000)
python -m uvicorn apps.api.main:app --host 127.0.0.1 --port 8000 --reload

# Terminal 2: Launch Durable Analytics Worker (Processes async jobs)
python -m analytics.worker
```

---

### Step 6: Run the Dashboard
```powershell
# Terminal 3: Launch Next.js Web Interface (Port 3000)
cd apps/dashboard
npm run dev
```
Open **[http://localhost:3000](http://localhost:3000)** in your browser. API documentation is available at **[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)**.

---

## 16. Environment Variables

All secrets and system variables are configured via `.env` in the repository root. Variables are never leaked to client bundles.

| Variable Name | Required | Default Value | Description |
| :--- | :---: | :--- | :--- |
| `DATABASE_URL` | No | `sqlite:///data/lineguard.db` | SQLAlchemy connection string. Set to `postgresql+psycopg://lineguard:local-demo-only@localhost:5432/lineguard` for Postgres. |
| `LINEGUARD_API_TOKEN` | No | *(None)* | Optional secret token. If configured, requires `Authorization: Bearer <token>` on all `/api/*` endpoints except `/api/health`. |
| `LINEGUARD_MODEL_DEVICE`| No | `cpu` | Device for PyTorch and PatchCore inference (`0` or `cuda:0` for GPU). |
| `LINEGUARD_PROXY_WEIGHTS`| No | *(active.json)* | Direct path override for YOLO11n weights file. |
| `NVIDIA_API_KEY` | No | *(None)* | NVIDIA NIM API key for inspection assistant and vision description. |
| `NVIDIA_MODEL` | No | `nvidia/nemotron-3-ultra-550b-a55b` | LLM model identifier for grounded evidence Q&A. |
| `NVIDIA_VISION_MODEL` | No | `meta/llama-3.2-11b-vision-instruct` | Multimodal VLM model for automated frame descriptions. |
| `ELEVENLABS_API_KEY` | No | *(None)* | ElevenLabs API key for spoken voice alerts and transcription. |
| `ELEVENLABS_VOICE_ID` | No | `EXAVITQu4vr4xnSDxMaL` | Voice ID for critical alert speech synthesis. |
| `SUPERMEMORY_API_KEY` | No | *(None)* | API key to store and search engineer-approved corrective actions. |

---

## 17. Interactive Demo Walkthrough

Follow this sequence to evaluate the platform during a live demonstration or judging review:

```
[1. Synthetic Critical Replay] ──> [2. Inspect Findings & RCA] ──> [3. 3D Line Halts]
                 │
                 ▼
[4. Spoken ElevenLabs Alert]  ──> [5. Engineer Sign-Off]       ──> [6. 20-Part Verification]
                 │
                 ▼
[7. Audit Hash Verification]  ──> [8. Process Lab What-If]    ──> [9. Live Camera Scan]
```

1. **Demonstrate Critical Replay:**
   - In the **Inspections** workspace, select the scenario **Thermal drift · critical crack** and click **Replay Scenario**.
   - Observe the highlighted surface crack bounding box, polar coordinates ($r=112\text{ mm}$, $\theta=45^\circ$), and deterministic **Critical** severity badge.
2. **Review Root Cause Analysis:**
   - Scroll to the **Process Context & RCA** panel.
   - Inspect the TreeSHAP feature contribution chart indicating that elevated temperature (+142°C drift) was the primary statistical driver of the defect.
3. **Verify Simulated Plant Interlock:**
   - Open the **Production Line** simulation panel.
   - Notice the Three.js manufacturing lifecycle automatically transitions to a pulsing red **QUALITY HOLD** state. Parts are halted at the inspection station.
   - If ElevenLabs credentials are configured, an audible spoken alert announces the critical defect and affected lot.
4. **Record Engineer Decision:**
   - Enter an engineer identity (e.g., `M. Sterling - Lead Quality Engineer`) and select **Approve Corrective Action**.
   - Confirm the decision is committed to database persistence. Attempting to approve again is blocked by optimistic concurrency locking.
5. **Execute 20-Part Evidence Verification:**
   - Click **Run 20-Part Verification**. The system replays 20 consecutive passed inspections from the same machine.
   - The status transitions to `demo_criteria_met` and computes the one-sided 95% upper defect rate ($1 - 0.05^{1/20} = 13.91\%$).
   - Click **Resume Production Line** to clear the interlock.
6. **Verify Cryptographic Audit Trail:**
   - Switch to the **Evidence & audit** tab.
   - Inspect the SHA-256 hash chain and click **Verify Audit Integrity** to recompute the rolling hash across all recorded events.
7. **Simulate Process Shifts in Process Lab:**
   - Navigate to the **Process lab** tab.
   - Adjust the **Temperature** and **Vibration** sliders.
   - Observe real-time shifts in TreeSHAP margin attributions and Monte Carlo next-lot defect predictions without modifying database records.
8. **Live Camera & AR Scanning:**
   - Switch to the **Camera station** tab, enable your webcam, and point it at a physical metal part or sample photo.
   - Test the real-time quality gate (observe blur rejection when moving the camera rapidly) and inspect the continuous PatchCore anomaly heatmap.

---

## 18. Testing & Quality Assurance

LineGuard includes 125+ automated unit tests, integration tests, and type-checking suites:

```powershell
# 1. Run Complete Backend Unit & Integration Suite
python -m unittest discover -s tests -v

# 2. Run Focused Test Suites
python -m unittest tests.test_pipeline -v           # End-to-end inspection pipeline
python -m unittest tests.test_patchcore -v          # PatchCore feature & coreset tests
python -m unittest tests.test_rca -v                # XGBoost & TreeSHAP analytics
python -m unittest tests.test_forecast -v           # PLSR & Monte Carlo forecasting
python -m unittest tests.test_quality_gate -v       # Laplacian blur & specularity tests
python -m unittest tests.test_backend -v            # FastAPI endpoints & audit verification

# 3. Run Frontend Type Checks
cd apps/dashboard
npm run typecheck

# 4. Verify Production Frontend Build
npm run build
```

---

## 19. Implementation Status

| System Domain | Status | Implemented Functionality | Pending / Future Scope |
| :--- | :---: | :--- | :--- |
| **Image Ingestion** | ✅ Complete | Bounded streaming, size caps ($\le 8\text{ MB}$), format decoding, SHA-256 hashing | Multi-camera synchronized genlock ingestion |
| **Quality Gate** | ✅ Complete | Laplacian blur variance, specular saturation filtering, profile baselines | Automated photometric lighting calibration |
| **Object Detection** | ✅ Complete | YOLO11n fine-tuned on NEU steel surfaces, SAHI tiled slicing | Annotated automotive brake-disc crack dataset |
| **Anomaly Detection**| ✅ Complete | PatchCore with ResNet-18 coreset memory banks on casting & MPDD | 3D surface point-cloud anomaly detection |
| **Corrosion Classifier**| ✅ Complete | Dual-head ResNet-18 corrosion scoring with color saturation gating | Cross-alloy metallurgical depth estimation |
| **Metrology** | 🟡 Partial | Polar coordinate mapping, contour bounding box, radial zones | Validated sub-millimeter camera lens calibration |
| **Severity Triage** | ✅ Complete | Deterministic rules v3 (class, physical size, zone, multiplicity) | Plant-specific OEM automotive engineering limits |
| **RCA Engine** | ✅ Complete | XGBoost multi-class classifier with exact TreeSHAP margin attributions | Real-world shop-floor causal identification |
| **Risk Forecasting** | ✅ Complete | PLSR forecaster, PCR benchmark, 10k Monte Carlo residual bootstrap | Multivariate sensor-fused machine failure modeling |
| **SOP & Actions** | ✅ Complete | Deterministic SOP mapping, part hold vs. lot quarantine, verification gate | Bidirectional PLC / OPC-UA fieldbus control |
| **Traceability** | ✅ Complete | Frame hashes, model versioning, append-only SHA-256 hash chain | Distributed blockchain / RFC 3161 timestamping |
| **Dashboard** | ✅ Complete | Material 3 interface, SVG charts, Three.js 3D line, Process Lab sandbox | Multi-tenant user role permission management |
| **Mobile & AR** | 🟡 Partial | 2D OpenCV.js silhouette tracking, sampled heatmap, Cloudflare tunnel | Validated 6-DoF SLAM surface AR tracking |
| **LLM & Speech** | ✅ Complete | Grounded NVIDIA Nemotron Q&A, ElevenLabs critical audio alerts | Local on-device edge LLM execution |

---

## 20. Known Limitations & Engineering Truths

To maintain rigorous technical credibility, the following architectural and empirical boundaries must be acknowledged:

1. **Proxy Datasets vs. Brake Discs:** The active vision models were trained on proxy datasets: NEU-DET (hot-rolled steel sheets), casting impeller photographs, and MPDD (sheet metal components). While the platform architecture and geometric metrology are designed for brake discs, **no performance guarantee on physical automotive brake discs is claimed**.
2. **Uncalibrated Model Scores:** PatchCore anomaly distances and XGBoost softmax outputs are uncalibrated feature distances and classifier probabilities. They do not represent empirical real-world failure probabilities and are deliberately barred from directly triggering critical safety decisions.
3. **Synthetic Telemetry:** Telemetry records (`temperature_c`, `pressure_bar`, `vibration_mm_s`, `machine_speed_rpm`) are generated from parameterized physical drift models. TreeSHAP values reflect statistical associations within these synthetic scenarios, not proven physical causality.
4. **2D Tracking vs. 6-DoF AR:** The browser camera view uses 2D OpenCV.js contour detection and Lucas-Kanade optical flow. It is a 2D visualization demo, not calibrated 6-DoF spatial computing with sub-millimeter surface registration.
5. **Verification Sample Limits:** The 20-part clean verification rule establishes that the observed sample defect rate is 0. Under the rule of three, 0 defects in 20 parts yields a one-sided 95% upper defect rate bound of $13.91\%$. It validates demo compliance, not a permanent production fix.
6. **Simulated Production Interlock:** The Three.js manufacturing line simulation halts automatically upon a critical finding. It does not transmit hardware interlock signals to industrial PLCs or safety relays.

---

## 21. Roadmap

- [x] Decoupled FastAPI backend and Next.js 16 App Router dashboard.
- [x] Calibrated pre-inference quality gate for blur and overexposure.
- [x] YOLO11n proxy detection with SAHI tiled slicing integration.
- [x] PatchCore unsupervised anomaly detection with ResNet-18 coreset banks.
- [x] ResNet-18 dual-head corrosion classifier with color gating.
- [x] Deterministic severity rules v3 separating ML confidence from safety triage.
- [x] XGBoost process classifier with exact TreeSHAP feature attributions.
- [x] PLSR next-lot defect forecaster and 10,000-sample Monte Carlo uncertainty simulator.
- [x] Append-only SHA-256 cryptographic audit ledger and verification endpoint.
- [x] Interactive Process Lab ("What-If") parameter sandbox.
- [x] Three.js procedural 3D manufacturing lifecycle with critical halt/resume.
- [x] NVIDIA Nemotron grounded assistant and ElevenLabs critical spoken alerts.
- [ ] Ingest and label genuine automotive brake-disc crack and porosity dataset.
- [ ] Connect live historical sensor time-series directly into the PLSR inference pipeline.
- [ ] Add dedicated grouped affected-batch quarantine management view (Track 3 Slide 17).
- [ ] Upgrade engineer assertions to signed JWT / OAuth2 role-based authentication.
- [ ] Implement industrial OPC-UA / MQTT adapter for direct PLC integration.

---

## 22. Security & Authentication

- **API Authentication:** Optional Bearer token validation can be enabled via `LINEGUARD_API_TOKEN`. When set, all mutation and inspection retrieval endpoints enforce constant-time digest comparison (`secrets.compare_digest`).
- **Credential Storage:** All third-party API credentials (NVIDIA, ElevenLabs, Supermemory) and database connection strings reside server-side in `.env` and are strictly excluded from client-facing Next.js bundles.
- **Audit Tamper-Resistance:** Audit records utilize SHA-256 forward-chaining. Modifying a historical record or inserting an unauthorized row invalidates the cryptographic verification check.
- **Mobile Security:** Temporary phone demo sessions utilize a 6-digit cryptographically random pairing code, rate-limited attempts, and signed HTTP-only session cookies over Cloudflare HTTPS tunnels.
- **Operator Identity:** Current engineer sign-offs are logged assertions. Production deployments require binding to enterprise Single Sign-On (SSO) before authorizing safety sign-offs.

---

## 23. Contributing

Contributions are welcome. Please adhere to the following workflow:

1. **Fork the Repository:** Create your own fork on GitHub.
2. **Create a Feature Branch:**
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Ensure Code Quality & Formatting:**
   - Adhere to PEP 8 standards for Python and ESLint/Prettier for TypeScript.
   - Maintain documentation integrity: do not delete architectural caveats or limitations.
4. **Execute the Verification Suite:**
   ```bash
   python -m unittest discover -s tests -v
   cd apps/dashboard && npm run typecheck && npm run build
   ```
5. **Commit with Clear Semantic Messages:**
   ```bash
   git commit -m "feat(vision): add calibrated lens distortion correction"
   ```
6. **Submit a Pull Request:** Open a PR against `main` with a clear explanation of changes and test evidence.

---

## 24. License

This project is licensed under the **Apache License 2.0**. See the [LICENSE](LICENSE) file for full license terms and conditions.

---

## 25. Acknowledgements & References

LineGuard builds upon foundational open-source research and engineering libraries:
- **PatchCore:** Roth et al., *Towards Total Recall in Industrial Anomaly Detection* (CVPR 2022). [Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Roth_Towards_Total_Recall_in_Industrial_Anomaly_Detection_CVPR_2022_paper.pdf) · [Amazon Science](https://github.com/amazon-science/patchcore-inspection).
- **YOLO11:** [Ultralytics](https://github.com/ultralytics/ultralytics) for real-time computer vision architectures.
- **SAHI:** Akyon et al., *Slicing Aided Hyper Inference and Fine-Tuning for Small Object Detection*. [GitHub](https://github.com/obss/sahi).
- **SHAP:** Lundberg and Lee, *A Unified Approach to Interpreting Model Predictions* (NeurIPS 2017). [GitHub](https://github.com/shap/shap).
- **NEU-DET:** Northeastern University surface defect database for hot-rolled steel strip.
- **MPDD:** Metal Parts Defect Detection dataset for anomaly detection benchmarks.
- **MVTec AD:** MVTec Software GmbH industrial anomaly detection benchmark.

---

## 26. Project Status & Engineering Disclaimer

> [!IMPORTANT]
> **Engineering Disclaimer:** LineGuard is a technology demonstration and research prototype developed for the **Singularity 2026 Hackathon (Track 3: Automotive Quality Inspection)**. It is **not certified** for safety-critical, commercial automotive manufacturing deployment. No safety-critical equipment, factory PLC, or automotive assembly line should be controlled without certified sensor calibration, hardware fail-safes, and regulatory compliance validation.
