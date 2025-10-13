# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Also, we use [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

### [2.2.1] - 2025-09-22

#### Added

- **🔧 Evaluation UI:** Add URL parameter control for Evaluation button visibility.

---

### [2.2.0] - 2025-09-09

#### Added

- **🧪 Skillcompanion-Evaluation:** Initial integration of Skillcompanion-Evaluation.

---

### [2.1.1] - 2025-09-02

#### Fixed

- **🛠️ CI:** Fix yaml error in CI pipeline.

#### Changed

- **🛠️ CI:** Use templates for docker build and cleanup stages. Use private pipeline runner.

---

### [2.1.0] - 2025-08-28

#### Added

- **🧪 Evaluation:** Script for Skillcompanion-Evaluation.

---

### [2.0.0] - 2025-08-27

#### Fixed

- **🐳 Docker:** Add variants folder to backend Dockerfile. AI-880
- **🐳 Docker:** Remove deprecated Dockerfile copy command. AI-880

#### Changed

- **⚠️ BREAKING:** Introduce whitelabeling in ASTK and usage of variants for front and backend.  
  BREAKING: Remove page functionality in favor of starting the right streamlit app parametrically. AI-880.

---

### [1.5.1] - 2025-08-26

#### Added

- **🌐 Provider Config:** Add openwebui to provider config to make OWUI models available via /info endpoint.

---

### [1.5.0] - 2025-08-25

#### Added

- **🌐 Provider Config:** Add openwebui to provider config to make OWUI models available via /info endpoint. AI-880

---

### [1.4.0] - 2025-08-22

#### Changed

- **🎨 Streamlit:** Upgrade streamlit to utilize fonts and themes.

---

### [1.3.2] - 2025-08-21

#### Fixed

- **🎨 Fonts:** Add missing font files.

---

### [1.3.1] - 2025-08-14

#### Refactored

- **🔄 Consistency:** AGENT_URL to BACKEND_URL for consistency.
- **🐳 Docker:** Refactor Dockerfile for multi-stage build and dynamic page setup.

---

### [1.3.0] - 2025-08-01

#### Added

- **🔗 Dynamic Links:** URL parameter handling for dynamic link generation.

---

### [1.2.0] - 2025-07-30

#### Added

- **🔑 OWUI Token:** OWUI-Token from user or default-user for model.

#### Changed

- **🧩 Langfuse:** Merge Langfuse-Traces from Langgraph-Agent.

---

### [1.1.0] - 2025-07-29

#### Added

- **🧑‍💻 Skillcompanion:** Add Skillcompanion with interrupt and questions and interrupt.

---

### [1.0.0] - 2025-07-24

#### Added

- **🧑‍💻 Agents:** Add agents of the old version.
- **🧑‍💻 Pages:** Add old version pages.
- **🆚 Comparison:** Add comparison of old version ASTK with new (Part 1).
- **🔐 OWUI Login:** Add OWUI login to ASTK.
- **🧑‍💻 Skillcompanion:** Add Skillcompanion (agent and page).

#### Changed

- **🧑‍💻 Skillcompanion:** Update changes to use Skillcompanion.
