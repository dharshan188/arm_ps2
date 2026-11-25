---
name: Industrial Vision Dark Mode
colors:
  surface: '#001428'
  surface-dim: '#001428'
  surface-bright: '#273b50'
  surface-container-lowest: '#000f20'
  surface-container-low: '#061d31'
  surface-container: '#0b2135'
  surface-container-high: '#172b40'
  surface-container-highest: '#22364c'
  on-surface: '#d1e4ff'
  on-surface-variant: '#bec7d4'
  inverse-surface: '#d1e4ff'
  inverse-on-surface: '#1e3247'
  outline: '#88929d'
  outline-variant: '#3f4852'
  surface-tint: '#96ccff'
  primary: '#96ccff'
  on-primary: '#003353'
  primary-container: '#00a6ff'
  on-primary-container: '#00395b'
  inverse-primary: '#00639a'
  secondary: '#45f0d2'
  on-secondary: '#00382f'
  secondary-container: '#00d3b7'
  on-secondary-container: '#005549'
  tertiary: '#c9bfff'
  on-tertiary: '#2e009c'
  tertiary-container: '#a090ff'
  on-tertiary-container: '#3400ac'
  error: '#ffb4ab'
  on-error: '#690005'
  error-container: '#93000a'
  on-error-container: '#ffdad6'
  primary-fixed: '#cee5ff'
  primary-fixed-dim: '#96ccff'
  on-primary-fixed: '#001d32'
  on-primary-fixed-variant: '#004a76'
  secondary-fixed: '#55fbde'
  secondary-fixed-dim: '#26dec2'
  on-secondary-fixed: '#00201b'
  on-secondary-fixed-variant: '#005045'
  tertiary-fixed: '#e5deff'
  tertiary-fixed-dim: '#c9bfff'
  on-tertiary-fixed: '#1a0063'
  on-tertiary-fixed-variant: '#441cc8'
  background: '#001428'
  on-background: '#d1e4ff'
  surface-variant: '#22364c'
typography:
  display-lg:
    fontFamily: Inter
    fontSize: 48px
    fontWeight: '700'
    lineHeight: 56px
    letterSpacing: -0.03em
  display-lg-mobile:
    fontFamily: Inter
    fontSize: 32px
    fontWeight: '700'
    lineHeight: 40px
    letterSpacing: -0.02em
  headline-xl:
    fontFamily: Inter
    fontSize: 36px
    fontWeight: '600'
    lineHeight: 44px
    letterSpacing: -0.02em
  headline-xl-mobile:
    fontFamily: Inter
    fontSize: 26px
    fontWeight: '600'
    lineHeight: 34px
    letterSpacing: -0.01em
  headline-md:
    fontFamily: Inter
    fontSize: 24px
    fontWeight: '600'
    lineHeight: 32px
    letterSpacing: -0.01em
  headline-sm:
    fontFamily: Inter
    fontSize: 20px
    fontWeight: '600'
    lineHeight: 28px
    letterSpacing: 0em
  body-lg:
    fontFamily: Inter
    fontSize: 16px
    fontWeight: '400'
    lineHeight: 24px
    letterSpacing: 0em
  body-md:
    fontFamily: Inter
    fontSize: 14px
    fontWeight: '400'
    lineHeight: 20px
    letterSpacing: 0em
  body-sm:
    fontFamily: Inter
    fontSize: 12px
    fontWeight: '400'
    lineHeight: 18px
    letterSpacing: 0.01em
  telemetry-lg:
    fontFamily: JetBrains Mono
    fontSize: 18px
    fontWeight: '600'
    lineHeight: 24px
    letterSpacing: -0.01em
  telemetry-md:
    fontFamily: JetBrains Mono
    fontSize: 13px
    fontWeight: '500'
    lineHeight: 18px
    letterSpacing: 0.02em
  telemetry-sm:
    fontFamily: JetBrains Mono
    fontSize: 11px
    fontWeight: '500'
    lineHeight: 14px
    letterSpacing: 0.05em
  label-caps:
    fontFamily: JetBrains Mono
    fontSize: 10px
    fontWeight: '700'
    lineHeight: 12px
    letterSpacing: 0.08em
rounded:
  sm: 0.125rem
  DEFAULT: 0.25rem
  md: 0.375rem
  lg: 0.5rem
  xl: 0.75rem
  full: 9999px
spacing:
  gutter: 1rem
  gutter-desktop: 1.5rem
  margin: 1rem
  margin-desktop: 2rem
  space-xs: 0.25rem
  space-sm: 0.5rem
  space-md: 1rem
  space-lg: 1.5rem
  space-xl: 2.5rem
---

## Brand & Style

This design system delivers a mission-critical, operations-grade interface engineered for real-time computer vision, automated sorting machinery, and industrial telemetry. 

### Brand Personality & Emotional Intent
- **Precision Operator Trust:** Crisp, telemetry-focused clarity designed to eliminate cognitive fatigue during continuous monitoring.
- **Cyber-Industrial Refinement:** Modern instrumentation meets clean edge-computing diagnostics—high-tech, sophisticated, and austere.
- **Restrained Cybernetic Energy:** Electric accents cut through deep obsidian fields with measured intent, avoiding decorative neon noise in favor of targeted optical routing.

### Visual Movement
The aesthetic synthesizes **Precision Minimal-Tech** and **Restrained Luminescence**:
- Monolithic midnight canvas layers anchoring sharp data tiles.
- Border-defined data bounding boxes mimicking heads-up targeting overlays.
- Sub-surface translucent glow effects localized to active camera feeds, status pulses, and critical operational triggers.
- Absence of vertical navigation sidebars to maximize horizontal camera feeds, inspection conveyor lanes, and real-time inference tables.

## Colors

The palette is tuned specifically for low-light control rooms and operator consoles, prioritizing high data contrast, optical wavelength distinction for multi-category item sorting, and zero eye fatigue.

### Surface Architecture
- **Base Canvas:** `#080D18` (deep abyss midnight)
- **Panel Base (Level 1 Surface):** `#101A2B` (dark navy slate)
- **Panel Raised (Level 2 Surface / Interactive Overlays):** `#16243A`
- **Panel Highlight (Hover / Selected States):** `#1F324E`

### Signal & Accent Roles
- **Primary Accent (`#00A6FF`):** Electric blue; designated for system focus, operational states, active tracking cursors, and primary actionable components.
- **Secondary Accent (`#00D4B8`):** Luminous teal; reserved for successful material classifications, validated throughput, purity thresholds, and positive telemetry states.
- **Tertiary Accent (`#7B61FF`):** Deep synthetic violet; allocated to specialized diagnostic metrics, infrared feeds, or system bypass modes.
- **Operational Alerts:**
  - Hazard / Divert Action: `#FF3B56` (high-vis crimson)
  - Degradation / Jam Warning: `#FFB020` (amber flare)

### Typography & Structure Tints
- **Text High-Contrast:** `#F0F5FF` (crisp white with faint glacial undertone)
- **Text Supporting:** `#8FA3BD` (calibrated slate gray for metadata and unselected indicators)
- **Technical Outlines:** `rgba(0, 166, 255, 0.15)` for quiescent states; `rgba(0, 166, 255, 0.45)` for focused inspection nodes.

## Typography

The typographic hierarchy separates structural and natural-language elements from computational readouts.

- **Primary UI Text (Inter):** Applied across all standard navigation nodes, titles, modal headers, documentation, and operational labels. Inter delivers high legibility across dense data tables and multi-column readouts.
- **Telemetry & Diagnostics (JetBrains Mono):** Mandated for object tracking IDs, real-time FPS counts, purity percentages, bounding box coordinates, sorting actuator latency metrics, and hardware statuses.
- **Case Rules:** Operational system labels and confidence indexes must utilize `label-caps` in full uppercase tracking to provide instantaneous readability amidst rapid video streams.

## Layout & Spacing

The interface functions as an operations command center, utilizing a **Full-Bleed Horizontal Control Structure** without persistent vertical sidebars.

### Spatial Model
- **No Sidebar Layout:** Navigation and master operational controls reside exclusively within a fixed, low-profile top utility bar (height: 56px) leaving the full horizontal screen available for wide camera pipelines and real-time conveyor feeds.
- **16-Column Desktop Grid:** Built on an expanded 16-column fluid structure on displays >= 1440px to accommodate dense side-by-side video feeds, real-time inference tables, and spectral breakdown panels.
- **Screen Breakpoints:**
  - **Desktop / Operation Monitor (>= 1280px):** 16-column grid; `gutter-desktop` (1.5rem); multi-pane split view (Live Feed 10 cols, Telemetry/Actuators 6 cols).
  - **Field Tablet (768px - 1279px):** 8-column layout; stacked feed above real-time data logs; `gutter` (1rem).
  - **Handheld Diagnostic (< 768px):** 4-column layout; full-width stacked cards; secondary streams collapse into tabbed interfaces; `margin` (1rem).

### Layout Rules
- Module interiors use uniform increments of `space-sm` (0.5rem) and `space-md` (1rem) for tight information architecture.
- Section boundaries and separate functional blocks require strict separation using `space-lg` (1.5rem) or `space-xl` (2.5rem).

## Elevation & Depth

Visual depth is achieved through **Surface Stacking, Sub-Surface Luminescence, and Precision Borders**, rather than traditional diffused drop shadows.

### Elevation Hierarchy
- **Canvas (`#080D18`):** Base coordinate plane. No elevation.
- **Panel Tier 1 (`#101A2B`):** Machine vision feeds, static metric groups, and telemetry data tables. Framed by a 1px border of `rgba(0, 166, 255, 0.12)`.
- **Panel Tier 2 (`#16243A`):** Interactive widgets, popover parameter adjusters, and active camera control overlays. Framed by a 1px border of `rgba(0, 166, 255, 0.25)` accompanied by a subtle ambient back-glow: `box-shadow: 0 0 20px rgba(0, 166, 255, 0.08)`.
- **Active Focus & Warning Elevation:**
  - Standard focus: `0 0 12px rgba(0, 166, 255, 0.35)` with outline `#00A6FF`.
  - Contaminant Alert: `0 0 16px rgba(255, 59, 86, 0.4)` with outline `#FF3B56`.

### Glass & Transparency
Floating operational heads-up modules overlaid on real-time video feeds must use `rgba(16, 26, 43, 0.82)` with backdrop filtering set to `backdrop-filter: blur(12px)`.

## Shapes

The interface employs a disciplined, industrial **Soft/Engineered Edge** (`roundedness: 1`), conveying structural integrity and mechanical calibration.

- **Base Radius (0.25rem / 4px):** Standard interactive elements, badges, form inputs, inspection chips, and bounding-box indicators.
- **Large Radius (0.5rem / 8px):** Structural video feed containers, diagnostic data cards, and telemetry overview panels (`rounded-lg`).
- **Extra-Large Radius (0.75rem / 12px):** Global operational modals, calibration configuration drawers, and full-screen diagnostics viewports (`rounded-xl`).
- **Precision Chamfers:** Diagnostic reticles, conveyor lane dividers, and video crosshairs maintain strict 90-degree squared corners (0px) to signify optical measurement accuracy.

## Components

### Buttons
- **Primary Action (Trigger/Engage):** Background `#00A6FF`, text `#080D18` (high-contrast deep black for punchy visibility), font weight 600. On hover: box shadow `0 0 16px rgba(0, 166, 255, 0.5)`.
- **Secondary (Telemetry/Acknowledge):** Background `rgba(0, 212, 184, 0.1)`, border 1px solid `#00D4B8`, text `#00D4B8`.
- **Subtle/Utility:** Background `#16243A`, border 1px solid `rgba(0, 166, 255, 0.15)`, text `#F0F5FF`.
- **Emergency Halt / Override:** Background `#FF3B56`, text `#FFFFFF`, glowing outline `0 0 20px rgba(255, 59, 86, 0.6)`.

### Chips & Classification Tags
- Composed of `label-caps` in JetBrains Mono with 0.25rem border-radius.
- **PET Clear:** Tinted `#00A6FF` background (`rgba(0, 166, 255, 0.12)`), text `#00A6FF`, border `rgba(0, 166, 255, 0.3)`.
- **HDPE Coloured:** Tinted `#00D4B8` background (`rgba(0, 212, 184, 0.12)`), text `#00D4B8`, border `rgba(0, 212, 184, 0.3)`.
- **Contaminant / Reject:** Tinted `#FF3B56` background (`rgba(255, 59, 86, 0.15)`), text `#FF3B56`, border `rgba(255, 59, 86, 0.4)`.

### Data Cards & Vision Containers
- Background `#101A2B`, border 1px solid `rgba(0, 166, 255, 0.15)`, radius 0.5rem.
- Include a recessed technical header strip featuring uppercase micro-labels (`label-caps`), live ping indicator, and hardware sensor telemetry.

### Form Inputs & Sensitivity Sliders
- **Input Fields:** Background `#080D18`, border 1px solid `rgba(143, 163, 189, 0.25)`, text `#F0F5FF`, font JetBrains Mono. Focus state: border `#00A6FF`, subtle glow `0 0 8px rgba(0, 166, 255, 0.3)`.
- **Sliders (Confidence Thresholds):** Track background `#16243A`, active fill `#00A6FF`, thumb 12px square or circular node with `#00A6FF` and trailing numerical tooltip.

### Checkboxes & Toggle Switches
- **Checkboxes:** Squared 16px boxes, border 1.5px solid `rgba(0, 166, 255, 0.3)`, fill `#080D18`. Checked state: fill `#00A6FF`, checkmark icon `#080D18`.
- **Toggles (Air Jets / Optical Sorters):** Track `#16243A` with active background `#00D4B8`. Handle `#F0F5FF` with a 2px offset.

### Domain-Specific Components
- **Inference Bounding Overlays:** Vector outlines on live feed canvas with 1px stroke (`#00A6FF` or `#00D4B8`), accompanied by floating JetBrains Mono anchor tag displaying `MATERIAL_CLASS: [CONFIDENCE %]`.
- **Throughput Rate Ticker:** Large monospaced display showing metric tons per hour (t/h) or objects per second (ops/s), accented with micro-sparkline charts illuminated in electric blue.