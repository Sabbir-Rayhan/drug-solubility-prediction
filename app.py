import streamlit as st
import joblib
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.Chem import rdFingerprintGenerator   # new import

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Drug Solubility Predictor",
    page_icon="🧪",
    layout="centered"
)

# ── Load model files ──────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model   = joblib.load("best_model_scaffold.pkl")
    scaler  = joblib.load("scaler_scaffold.pkl")
    encoder = joblib.load("label_encoder_scaffold.pkl")
    return model, scaler, encoder

model, scaler, encoder = load_model()

# ── Feature extraction — MUST match notebook exactly (525 features) ───────────
def extract_features(smiles_str):
    mol = Chem.MolFromSmiles(smiles_str)
    if mol is None:
        return None
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return None

    # 13 physicochemical descriptors
    features = [
        Descriptors.MolWt(mol),
        Descriptors.MolLogP(mol),
        Descriptors.TPSA(mol),
        Descriptors.NumHAcceptors(mol),
        Descriptors.NumHDonors(mol),
        Descriptors.MolMR(mol),
        mol.GetNumHeavyAtoms(),
        rdMolDescriptors.CalcNumHeteroatoms(mol),
        rdMolDescriptors.CalcNumRotatableBonds(mol),
        rdMolDescriptors.CalcNumAromaticRings(mol),
        rdMolDescriptors.CalcNumRings(mol),
        Descriptors.LabuteASA(mol),
        Descriptors.BertzCT(mol),
    ]

    # ECFP4 fingerprint (radius=2, 256 bits) using modern generator
    gen4 = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=256)
    fp4 = gen4.GetFingerprintAsNumPy(mol)   # returns numpy array of 0/1 ints
    features.extend(int(b) for b in fp4)    # explicit int conversion for safety

    # ECFP6 fingerprint (radius=3, 256 bits) using modern generator
    gen6 = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=256)
    fp6 = gen6.GetFingerprintAsNumPy(mol)   # returns numpy array of 0/1 ints
    features.extend(int(b) for b in fp6)

    return features   # 525 total features

# ── UI ────────────────────────────────────────────────────────────────────────
st.title("🧪 Drug Solubility Prediction")
st.markdown(
    "Predicts aqueous solubility class (High / Medium / Low) "
    "using a scaffold-split XGBoost model trained on AqSolDB."
)
st.markdown("---")

# Example SMILES for quick testing
st.markdown("**Quick test — click an example:**")
col1, col2, col3 = st.columns(3)
example = ""
with col1:
    if st.button("Aspirin (High)"):
        example = "CC(=O)Oc1ccccc1C(=O)O"
with col2:
    if st.button("Caffeine (Medium)"):
        example = "Cn1cnc2c1c(=O)n(c(=O)n2C)C"
with col3:
    if st.button("Ibuprofen (Low)"):
        example = "CC(C)Cc1ccc(cc1)C(C)C(=O)O"

smiles_input = st.text_input(
    "Enter SMILES String",
    value=example,
    placeholder="e.g. CC(=O)Oc1ccccc1C(=O)O"
)

st.markdown("---")

if st.button("🔬 Predict Solubility", use_container_width=True):
    if not smiles_input.strip():
        st.warning("Please enter a SMILES string.")
    else:
        feats = extract_features(smiles_input.strip())

        if feats is None:
            st.error("❌ Invalid SMILES string. Please check and try again.")
        else:
            feats_scaled = scaler.transform([feats])
            pred_enc     = model.predict(feats_scaled)
            pred_proba   = model.predict_proba(feats_scaled)[0]
            pred_label   = encoder.inverse_transform(pred_enc)[0]
            confidence   = np.max(pred_proba) * 100

            # ── Result box ────────────────────────────────────────────────
            st.markdown("### 🎯 Prediction Result")
            if pred_label == "High":
                st.success(f"✅ **High Solubility**  —  Confidence: {confidence:.1f}%")
                st.markdown(
                    "> The compound is likely to dissolve well in water (LogS > 0). "
                    "Suitable for oral drug formulations."
                )
            elif pred_label == "Medium":
                st.warning(f"⚠️ **Medium Solubility**  —  Confidence: {confidence:.1f}%")
                st.markdown(
                    "> Moderate aqueous solubility (−2 < LogS ≤ 0). "
                    "May require formulation optimization."
                )
            else:
                st.error(f"❌ **Low Solubility**  —  Confidence: {confidence:.1f}%")
                st.markdown(
                    "> Poor aqueous solubility (LogS ≤ −2). "
                    "Likely to have bioavailability challenges."
                )

            # ── Probability bar chart ──────────────────────────────────────
            st.markdown("### 📊 Class Probabilities")
            classes = encoder.classes_
            prob_df = {cls: round(float(p) * 100, 2)
                       for cls, p in zip(classes, pred_proba)}

            for cls, pct in sorted(prob_df.items(),
                                   key=lambda x: x[1], reverse=True):
                color = {"High": "🟢", "Medium": "🟡", "Low": "🔴"}.get(cls, "⚪")
                st.markdown(f"{color} **{cls}**: {pct:.1f}%")
                st.progress(pct / 100)

            # ── Molecule info ──────────────────────────────────────────────
            mol = Chem.MolFromSmiles(smiles_input.strip())
            st.markdown("### 🔬 Molecular Properties")
            c1, c2, c3 = st.columns(3)
            c1.metric("Mol. Weight",  f"{Descriptors.MolWt(mol):.2f} g/mol")
            c2.metric("LogP",         f"{Descriptors.MolLogP(mol):.2f}")
            c3.metric("TPSA",         f"{Descriptors.TPSA(mol):.2f} Å²")
            c4, c5, c6 = st.columns(3)
            c4.metric("H-Acceptors",  Descriptors.NumHAcceptors(mol))
            c5.metric("H-Donors",     Descriptors.NumHDonors(mol))
            c6.metric("Rotatable Bonds", rdMolDescriptors.CalcNumRotatableBonds(mol))

# ── Footer ─────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "Model: XGBoost · Split: Bemis-Murcko Scaffold · "
    "Balancing: SMOTE · Features: 525 (ECFP4 + ECFP6 + 13 descriptors) · "
    "Dataset: AqSolDB (9,982 compounds)"
)
