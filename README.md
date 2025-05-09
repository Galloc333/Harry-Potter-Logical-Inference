# 🧙 Harry Potter Logical Inference Project

This repository contains an implementation of logical inference algorithms applied in a Harry Potter-themed scenario, specifically designed for navigating and safely exploring the Gringotts Bank environment.

## 📖 Overview
Harry Potter must strategically navigate a dangerous grid-based environment within Gringotts Bank, aiming to locate the Deathly Hallow hidden in one of several vaults. Along the way, he must:
- Avoid stepping onto tiles containing dragons or traps.
- Logically infer the location of traps through environmental clues (sulfur smell).
- Destroy traps strategically when necessary.
- Find and collect the Deathly Hallow.

The project demonstrates practical use of logical inference techniques and knowledge-base management in uncertain environments.

## 🎯 Project Goals
- Efficiently implement logical inference methods (propositional logic).
- Dynamically build and update a knowledge base from partial observations.
- Safely navigate and make informed decisions under uncertainty.

## 🚀 Features
- Logical inference algorithms (propositional logic and inference).
- Dynamic environment exploration with real-time updates.
- Strategic action selection: move, wait, destroy traps, and collect objects.

## 🗃️ Files & Structure
- `ex2.py`: Main implementation of the `GringottsController` logic.
- `checker.py`: Automated checking and validation of implemented logic.
- `utils.py`: Utility functions supporting the inference and logical reasoning.
- `inputs.py`: Sample inputs for testing and validation.
- `Homework_2.pdf`: Original problem statement and description.

## 🛠️ How to Run
Clone the repository:
```bash
git clone https://github.com/<YourUsername>/Harry-Potter-Logical-Inference.git
cd Harry-Potter-Logical-Inference
python checker.py
