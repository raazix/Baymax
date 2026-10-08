# LineGuard
<!-- impeccable:product-schema 1 -->

## Platform
web

## Stack
Next.js + TypeScript, explicitly selected by user; FastAPI, SQLAlchemy; SQLite local development and PostgreSQL option.

## Users
Quality engineers review component defects and approve corrective actions. Hackathon judges evaluate a working Track 3 prototype.

## Product Purpose
Detect, locate, explain and anticipate automotive component defects, starting with brake discs.

## Operating Context
Singularity 2026 Track 3. User has a webcam and RTX 4050 laptop, no brake-disc image collection, and 18 hours remaining at project kickoff.

## Capabilities and Constraints
Single-screen inspection image, type, severity, probable cause, future risk, action and batch alert. Synthetic data is permitted by the supplied brief. Separate model detection from engineering severity. Human approval precedes corrective action; no PLC override. Every decision links to evidence. Initial replay is synthetic; trained model accuracy and real production validation are unavailable.

## Evidence on Hand
Singularity_2026_Hackathon_Problem_Statements.pptx, slides 15–19. User's master architecture. No real images or paired production telemetry.

## Product Principles
Evidence reconstructs decisions. Model explanations are hypotheses, not causal proof. Simulated results must remain visibly labelled. Forecast validation must prevent temporal and batch leakage.

## Open Decisions
Deployment target and production engineering thresholds remain undecided.
