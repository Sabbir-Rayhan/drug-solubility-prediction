<div align="center">

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=700&size=28&pause=1000&color=00D4FF&center=true&vCenter=true&width=700&lines=Drug+Solubility+Prediction;Scaffold-Aware+Machine+Learning;Cheminformatics+%2B+XGBoost+%2B+SHAP" alt="Typing SVG" />

<br/>

[![Live App](https://img.shields.io/badge/🌐_Live_App-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://drug-solubility-prediction-hpluq2ynhame4vpvyeuygn.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-Model-EA580C?style=for-the-badge)](https://xgboost.readthedocs.io)
[![RDKit](https://img.shields.io/badge/RDKit-Cheminformatics-2E8B57?style=for-the-badge)](https://www.rdkit.org)
[![IEEE](https://img.shields.io/badge/IEEE-Conference_Paper-00629B?style=for-the-badge&logo=ieee)](https://ieee.org)

<br/>

> 🔬 **Predicts whether a drug compound is High / Medium / Low soluble in water**
> using a scaffold-aware XGBoost pipeline trained on 9,982 real pharmaceutical compounds.

<br/>

**⭐ If you find this useful, please star the repository!**

</div>

---

## 🎯 What This Project Does

This project builds a machine learning system that classifies the **aqueous solubility** of drug molecules — one of the most critical properties in pharmaceutical development. A drug that cannot dissolve in water cannot be absorbed by the body, making solubility prediction essential before costly lab synthesis.

Given a molecular SMILES string (a text representation of a molecule), the system predicts:

```
Aspirin  →  🟢 High Solubility   (98.3% confidence)
Caffeine →  🟡 Medium Solubility (98.8% confidence)
Ibuprofen → 🔴 Low Solubility   (97.9% confidence)
```

### 🌐 Try It Live → [drug-solubility-prediction.streamlit.app](https://drug-solubility-prediction-hpluq2ynhame4vpvyeuygn.streamlit.app/)

---

## 🚀 Novel Contributions (IEEE Conference Paper)

This project goes beyond standard ML benchmarks with **5 novel contributions**:

<table>
<tr>
<td width="50%">

### ✅ 1. Scaffold-Based Split
Instead of a random 80/20 split, molecules are split by their **Bemis-Murcko scaffold** — the core ring structure. This ensures the test set contains *structurally novel* molecules never seen during training.

**Finding:** Random splits overestimate accuracy by **3.91%** and ROC-AUC by **5.62%**

</td>
<td width="50%">

### ✅ 2. SMOTE Class Balancing
The dataset is severely imbalanced — only **10.6%** of compounds have High solubility. Without correction, the model ignores this minority class entirely.

SMOTE synthetically generates minority-class samples, improving High-class recall from **44% → significantly better** on the training distribution.

</td>
</tr>
<tr>
<td>

### ✅ 3. Dual Fingerprints (ECFP4 + ECFP6)
Previous work used ECFP4 only (captures 4-bond neighborhood). This work adds **ECFP6** (6-bond neighborhood), encoding broader molecular ring environments.

SHAP analysis confirms ECFP6 bits rank **higher** than ECFP4 bits in importance.

</td>
<td>

### ✅ 4. Extended Descriptors
Feature set expanded from **5 → 13 physicochemical descriptors**, adding MolMR, HeavyAtomCount, NumHeteroatoms, NumRotBonds, NumAromaticRings, RingCount, LabuteASA, BertzCT.

Total features: **261 → 525** (2.01× increase)

</td>
</tr>
<tr>
<td colspan="2">

### ✅ 5. SHAP Explainability
SHAP (SHapley Additive exPlanations) identifies *which molecular features* drive each prediction — making the model interpretable to chemists, not just a black box.

**Top predictor: LogP** (lipophilicity), consistent with known chemistry — hydrophobic molecules dissolve poorly in water.

</td>
</tr>
</table>

---

## 📊 Results at a Glance

### Model Comparison

| Model | Accuracy | F1-Score | ROC-AUC | Split Type |
|:------|:--------:|:--------:|:-------:|:----------:|
| XGBoost | 0.8397 | 0.8369 | 0.9420 | Random *(inflated)* |
| Random Forest | 0.8287 | 0.8228 | 0.9376 | Random *(inflated)* |
| **XGBoost + SMOTE** | **0.8006** | **0.7957** | **0.8858** | **Scaffold ✅ honest** |
| RF + SMOTE | 0.8066 | 0.7957 | 0.8770 | Scaffold ✅ honest |

### 🔑 The Generalization Gap — Key Finding

| Metric | Random Split | Scaffold Split | Overestimation |
|:-------|:------------:|:--------------:|:--------------:|
| Accuracy | 83.97% | 80.06% | **+3.91%** |
| F1-Score | 0.8369 | 0.7957 | **+4.12%** |
| ROC-AUC | 0.9420 | 0.8858 | **+5.62%** |
| High-class Recall | 67% | 44% | **+23 pts** |

> 💡 **Insight:** The 23-point drop in High-class recall under scaffold split reveals that random-split models partially *memorize* scaffold-specific patterns rather than learning true physicochemical principles. This is the paper's core finding.

---

## 🏗️ Project Architecture

```
drug-solubility-prediction/
│
├── 🐍 app.py                        # Streamlit web application (real-time prediction)
│
├── 🤖 best_model_scaffold.pkl        # Trained XGBoost model
│                                     # (scaffold split + SMOTE, 525 features)
├── ⚖️  scaler_scaffold.pkl            # StandardScaler (fitted on train only)
├── 🏷️  label_encoder_scaffold.pkl     # LabelEncoder (High / Medium / Low)
│
├── 📋 requirements.txt               # Python package dependencies
├── 📦 packages.txt                   # System dependencies (libxrender1 for RDKit)
└── 📖 README.md                      # This file
```

---

## 🔬 Technical Pipeline

```
Raw SMILES String
       │
       ▼
  ┌─────────────────────────────────────────┐
  │           RDKit Processing              │
  │  • Molecule validation & sanitization   │
  │  • 13 physicochemical descriptors       │
  │  • ECFP4 fingerprint  (256 bits)        │
  │  • ECFP6 fingerprint  (256 bits)  NEW   │
  └──────────────────┬──────────────────────┘
                     │  525 features
                     ▼
  ┌─────────────────────────────────────────┐
  │         Bemis-Murcko Scaffold Split     │
  │  • Group molecules by core scaffold     │
  │  • Train / Test with ZERO overlap       │
  └──────────────────┬──────────────────────┘
                     │
                     ▼
  ┌─────────────────────────────────────────┐
  │         SMOTE Oversampling              │
  │  • Applied to training set ONLY         │
  │  • High class: 840 → 4,600 samples      │
  └──────────────────┬──────────────────────┘
                     │
                     ▼
  ┌─────────────────────────────────────────┐
  │     XGBoost Classifier                  │
  │  n_estimators=200, max_depth=7          │
  │  learning_rate=0.1, subsample=0.8       │
  └──────────────────┬──────────────────────┘
                     │
                     ▼
        High 🟢 / Medium 🟡 / Low 🔴
          + Confidence Score (%)
          + Per-class Probabilities
          + 6 Molecular Properties
```

---

## ⚙️ Installation & Usage

### Run the App Locally

```bash
# Clone the repository
git clone https://github.com/Sabbir-Rayhan/drug-solubility-prediction.git
cd drug-solubility-prediction

# Install dependencies
pip install streamlit rdkit xgboost scikit-learn joblib numpy

# Launch
streamlit run app.py
```

### Use as a Python Library

```python
import joblib, numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.Chem import rdFingerprintGenerator

def predict(smiles):
    model   = joblib.load("best_model_scaffold.pkl")
    scaler  = joblib.load("scaler_scaffold.pkl")
    encoder = joblib.load("label_encoder_scaffold.pkl")

    mol = Chem.MolFromSmiles(smiles)

    # 13 descriptors + ECFP4 + ECFP6 = 525 features
    feats = [
        Descriptors.MolWt(mol), Descriptors.MolLogP(mol),
        Descriptors.TPSA(mol), Descriptors.NumHAcceptors(mol),
        Descriptors.NumHDonors(mol), Descriptors.MolMR(mol),
        mol.GetNumHeavyAtoms(),
        rdMolDescriptors.CalcNumHeteroatoms(mol),
        rdMolDescriptors.CalcNumRotatableBonds(mol),
        rdMolDescriptors.CalcNumAromaticRings(mol),
        rdMolDescriptors.CalcNumRings(mol),
        Descriptors.LabuteASA(mol), Descriptors.BertzCT(mol),
    ]
    gen4 = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=256)
    gen6 = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=256)
    feats += gen4.GetFingerprintAsNumPy(mol).tolist()
    feats += gen6.GetFingerprintAsNumPy(mol).tolist()

    pred  = encoder.inverse_transform(model.predict(scaler.transform([feats])))[0]
    conf  = max(model.predict_proba(scaler.transform([feats]))[0]) * 100
    return pred, conf

# Try it
print(predict("CC(=O)Oc1ccccc1C(=O)O"))   # Aspirin
# → ('High', 98.3)
```

---

## 🧠 SHAP Feature Importance

The top 5 features driving predictions:

| Rank | Feature | Chemical Meaning | SHAP Score |
|:----:|:--------|:----------------|:----------:|
| 🥇 1 | **LogP** | Lipophilicity — hydrophobic ↔ hydrophilic | ██████████ 3.7 |
| 🥈 2 | **BertzCT** | Molecular complexity | ██░░░░░░░░ 0.7 |
| 🥉 3 | **MolWt** | Molecular size | █░░░░░░░░░ 0.6 |
| 4 | **ECFP6_100** | Broad structural context (ring system) | ░░░░░░░░░░ 0.3 |
| 5 | **MolMR** | Molar refractivity (polarizability) | ░░░░░░░░░░ 0.2 |

---

## 📦 Dependencies

| Package | Version | Purpose |
|:--------|:-------:|:--------|
| `streamlit` | latest | Web application framework |
| `rdkit` | latest | Molecular processing, fingerprints |
| `xgboost` | latest | Gradient boosting classifier |
| `scikit-learn` | latest | Preprocessing, metrics |
| `imbalanced-learn` | latest | SMOTE oversampling |
| `shap` | latest | Model explainability |
| `numpy` | latest | Numerical computing |
| `pandas` | latest | Data manipulation |

---

## 📖 Dataset

**AqSolDB** — Curated Reference Set of Aqueous Solubility

| Property | Value |
|:---------|:------|
| Source | Sorkun et al., *Scientific Data*, 2019 |
| Compounds | 9,982 unique organic molecules |
| Target | LogS (log molar solubility) |
| Classes | High (10.6%) / Medium (28.5%) / Low (61.0%) |
| Features used | SMILES strings → RDKit computed |

---

## 📄 Research Paper

This project is submitted as an IEEE conference paper:

> **"Scaffold-Aware Drug Solubility Classification: Quantifying Generalization Gaps in Machine Learning Models with SMOTE and Extended Molecular Fingerprints on AqSolDB"**
>
> *Sabbir Rayhan et al. — IEEE Conference 2026 (Under Review)*

```bibtex
@inproceedings{rayhan2026scaffold,
  title   = {Scaffold-Aware Drug Solubility Classification},
  author  = {Rayhan, Sabbir and others},
  year    = {2026},
  note    = {IEEE Conference (Under Review)}
}
```

---

## 🤝 Contributing

Contributions, issues and feature requests are welcome!

1. Fork the repository
2. Create your feature branch: `git checkout -b feature/YourFeature`
3. Commit your changes: `git commit -m 'Add YourFeature'`
4. Push to the branch: `git push origin feature/YourFeature`
5. Open a Pull Request

---

---

<div align="center">

### ⭐ Star this repo if it helped you!

*Built using Python · RDKit · XGBoost · Streamlit · SHAP*

</div>
