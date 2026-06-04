# GenRe_v3: Constrained Recourse Mechanics

This document explains how **GenRe_v3** enforces financial constraints and searches for actionable recourse paths during the generation process.

> Note: This process might not be mathematically solid but logically applied as of now.

## 1. Constraint Enforcement (Logit Masking)

Unlike traditional counterfactual methods that rely on post-hoc clipping, GenRe_v3 enforces constraints **during the generative process** at the Transformer's output heads.

### How it works:
1.  **Feature Filtering**: The system first checks if a feature is actually used by the classifier (`features_used_by_model`). If a feature is not used (e.g., metadata or columns with zero weights in the ANN), constraints are **not** enforced, allowing the model to stay on the data manifold without unnecessary restriction.
2.  **Logit Masking**: For features used by the model:
    *   **Continuous Features**: The UI provides a range $[min, max]$. We compare the centers of all available bins for that feature against this range. Bins falling outside the range are assigned a logit value of `-∞` (implemented as `-1e9`).
    *   **Categorical Features**: The UI provides a list of "Allowed Values". Any category index not in this list is masked with `-1e9`.
3.  **Softmax Sampling**: When the `softmax` is applied to the masked logits, the probability of selecting an invalid bin or category becomes zero. This guarantees that the sampled output **strictly** adheres to the user's boundaries.

## 2. The Search Strategy (Manifold Optimization)

The recourse generation follows a **Sample-and-Optimize** strategy rather than a single-shot generation.

### The Algorithm:
1.  **Contextual Conditioning**: The factual state (the rejected user's profile) is encoded into the Transformer's memory.
2.  **Stochastic Exploration**: The `generate_demo_recourse` function runs multiple iterations ($k=150$). In each iteration:
    *   The Transformer generates a complete feature set (all 81 features) one by one.
    *   Constraints are applied at each step.
    *   A temperature parameter ($temp=0.5$) allows for diverse sampling, exploring different "actionable paths" within the same constraints.
3.  **Proxy Evaluation**: Each generated path is immediately passed through the `FlexibleANNProxy`.
4.  **Best-Path Selection**: The system tracks the path that yields the highest approval probability. If a path reaches "High Certainty" (>95% probability), the search short-circuits to save compute.

## 3. Simple Demo Flow

Imagine a user is rejected because of a high **DTI (Debt-to-Income)** ratio.

1.  **Factual State**: DTI = 45%.
2.  **User Constraint**: The user specifies they can only reduce DTI to a minimum of 35% (maybe they can't pay off more debt).
3.  **The Mask**: The generator sees 100 possible bins for DTI. It masks out all bins representing values $< 35\%$.
4.  **Sampling**: The model samples a new DTI (e.g., 38%) that is both **realistic** (frequently seen in successful loans) and **valid** (within the 35%–45% range).
5.  **Correlated Changes**: Because it's a Transformer, when it picks DTI = 38%, it automatically adjusts the next features (like `revol_util` or `annual_inc`) to be statistically consistent with that new DTI, ensuring the advice isn't just valid, but **feasible**.
6.  **Result**: The UI shows a set of recommendations that, when combined, move the user from $P(Approve)=20\%$ to $P(Approve)=65\%$.
