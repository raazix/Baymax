---
name: LineGuard
description: Evidence-linked component inspection and quality decisions
colors:
  primary: "#67509b"
  primary-container: "#eaddff"
  on-primary-container: "#291348"
  neutral-bg: "#fdf7ff"
  surface-container: "#f2ecf7"
  surface-high: "#eae3f0"
  outline: "#cec3d9"
  ink: "#241e2c"
  muted: "#61566c"
  error: "#9f3029"
typography:
  display:
    fontFamily: "Bricolage Grotesque Variable, sans-serif"
    fontSize: "34px"
    fontWeight: 600
    lineHeight: 1.15
  body:
    fontFamily: "Bricolage Grotesque Variable, sans-serif"
    fontSize: "16px"
    fontWeight: 400
    lineHeight: 1.55
rounded:
  sm: "12px"
  md: "20px"
  lg: "24px"
  xl: "28px"
spacing:
  sm: "8px"
  md: "16px"
  lg: "24px"
components:
  button-primary:
    backgroundColor: "{colors.primary}"
    textColor: "#ffffff"
    rounded: "999px"
    padding: "12px 20px"
  workspace-card:
    backgroundColor: "{colors.surface-container}"
    rounded: "{rounded.xl}"
    padding: "24px"
---

# Design System: LineGuard

## Overview

LineGuard is a calm, evidence-first quality station. Its visual character follows Android Material 3: soft purple surfaces, rounded tactile controls, and readable evidence arranged around the inspected component. Bricolage Grotesque gives headings and controls a distinctive voice while keeping dense manufacturing details approachable.

The experience moves from capture, to model evidence, to engineer review. Purple signals selection and primary action; red and amber remain reserved for quality severity and holds. The design makes provenance and simulated data legible so engineers can see what supports each finding.

**Key characteristics:** clear; tactile; evidence-led; responsive.

## Colors

A restrained purple palette sets the working surface; severity colors retain their established meaning.

### Primary
- **Material Purple** (#67509b): Primary actions and selected controls.
- **Lavender Container** (#eaddff): Selected surfaces and soft purple accents.

### Neutral
- **Lavender White** (#fdf7ff): Main workspace.
- **Soft Lilac** (#f2ecf7): Cards, navigation, and grouped controls.
- **Ink** (#241e2c): Main text.
- **Muted Violet** (#61566c): Secondary text.
- **Lilac Outline** (#cec3d9): Fields and dividers.
- **Error Red** (#9f3029): Critical/high status and production holds.

**The Severity Color Rule.** Purple marks workflow; red and amber continue to identify quality state.

## Typography

**Display and body font:** Bricolage Grotesque Variable with a sans-serif fallback.

**Character:** Warm, expressive headings meet clear operational reading. Body text starts at 16px; supporting text and control labels stay at 14px or larger.

### Hierarchy
- **Headline** (600, 34px, 1.15): Page title.
- **Title** (600, 22px, 1.3): Major evidence sections.
- **Component title** (600, 20px): Capture and card headings.
- **Body** (400, 16px, 1.55): Evidence and explanatory text.
- **Label** (500, 14?15px): Controls, metadata, and status chips.

## Layout

The desktop workspace uses an inset navigation drawer, a prominent component image, and a supporting evidence and action column. Panels expand in place. On narrow screens the evidence stacks into a single column and a fixed bottom navigation keeps inspection actions close. Camera and AR views use the screen as a full-bleed stage with a legible floating control sheet.

## Elevation & Depth

The system layers lavender surfaces with soft, offset shadows. A restrained white gloss along panel tops adds depth without reducing text contrast. Primary buttons receive a slightly stronger shadow to read as tactile actions.

### Shadow Vocabulary
- **Working panels:** `0 8px 24px rgba(53, 30, 78, .08), 0 2px 6px rgba(53, 30, 78, .04)` with a subtle inset white edge.
- **Primary actions:** `0 5px 12px rgba(62, 36, 99, .20)` with a soft white top highlight.
- **AR control sheet:** `0 16px 40px rgba(15, 9, 23, .25)`.

## Shapes

Panels use 24?28px corners; media uses 20px; fields use 14px; compact status labels use 10?12px. Primary buttons are pill shaped. Surface grouping uses gentle tonal change and open spacing instead of dense borders.

## Components

### Buttons
- **Primary:** Purple pill with white text, 12px 20px padding and a soft raised edge.
- **Secondary:** Outlined or tonal pill for supporting actions.
- **Hover / focus:** Short color and elevation transitions; keyboard focus uses a visible purple outline.

### Cards and containers
- **Shape:** 24?28px corners.
- **Surface:** Lavender container over lavender-white workspace.
- **Shadow:** Offset soft shadow and a subtle top gloss.
- **Content:** Keep defect, measurement, provenance, and next action together.

### Inputs and fields
- 48px minimum control height, 14px radius, lilac-tinted fill, and clear focus ring.

### Inspection and hold
- The image and model overlay are the visual anchor. High and critical findings use a red hold banner; production simulation stays held until engineer approval and explicit resume.
- AR overlays follow the camera target and hide when tracking registration is lost. AI part descriptions are tentative and remain separate from severity and disposition.

## Do's and Don'ts

- Keep body copy comfortably readable; avoid metadata below 14px.
- Keep evidence provenance and synthetic/proxy limitations visible at the decision point.
- Use purple for workflow and selected states; use severity colors only for quality state.
- Use shadows and gloss softly; preserve clear text and edge contrast.
- Keep mobile scanning and evidence review reachable without horizontal scrolling.
