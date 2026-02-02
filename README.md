# Perspectival Auditing: Structural Invariance Tool (v2.3)

This repository contains the official implementation and dataset for the paper:  
**"Perspectival Auditing: Testing Structural Invariance of Moral Reasoning in Large Language Models"** (Submitted to TMLR).

## 🔬 Project Overview
`Pos.py` is a formal auditing tool designed to measure **Topological Invariance** in Large Language Models. Unlike standard benchmarks that measure normative agreement, this tool extracts latent moral primitives—**$\langle V, C, S, Q, A \rangle$**—to test whether a model's internal causal world-model remains stable across competing philosophical frameworks (Utilitarianism, Deontology, and Virtue Ethics).

### Key Features:
* **Deterministic Auditing:** Optimized for $T=0.0$ inference via the Groq API.
* **Formal Verification:** Implements Logical Integrity Constraints (LICs) to detect structural incoherence.
* **Multi-Framework Mapping:** Systematic transformation of 84 scenarios across 3 ethical perspectives (504 total codes).

---

## 🚀 Setup & Installation

### Prerequisites
* Python 3.8+
* [Groq API Key](https://console.groq.com/keys)

### 1. Clone & Install
```bash
git clone [https://github.com/squarethepyramid/POS](https://github.com/squarethepyramid/POS)
cd POS
pip install -r requirements.txt
