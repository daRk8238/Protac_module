import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from rdkit import Chem, DataStructs
from rdkit.Chem import Draw, rdFingerprintGenerator

from database.schema import init_db, get_session, Bioactivity


# -----------------------------
# Setup
# -----------------------------
OUTPUT_DIR = "models/outputs/visualizations"
os.makedirs(OUTPUT_DIR, exist_ok=True)

engine = init_db("protac_designer.db")
session = get_session(engine)

print("[Viz] Starting visualization pipeline...")


# -----------------------------
# 1️⃣ Single Molecule View
# -----------------------------
top_activity = (
    session.query(Bioactivity)
    .order_by(Bioactivity.pchembl_value.desc())
    .first()
)

mol = Chem.MolFromSmiles(top_activity.compound.smiles)
img = Draw.MolToImage(mol, size=(400, 400))
img.save(f"{OUTPUT_DIR}/top_molecule.png")
print("✓ Saved top_molecule.png")


# -----------------------------
# 2️⃣ Top 6 Grid
# -----------------------------
top6 = (
    session.query(Bioactivity)
    .order_by(Bioactivity.pchembl_value.desc())
    .limit(20)
    .all()
)

mols = []
legends = []

for a in top6:
    m = Chem.MolFromSmiles(a.compound.smiles)
    if m:
        mols.append(m)
        legends.append(f"{a.pchembl_value:.2f}")

grid = Draw.MolsToGridImage(
    mols,
    molsPerRow=3,
    subImgSize=(300, 300),
    legends=legends
)

grid.save(f"{OUTPUT_DIR}/top20_grid.png")
print("✓ Saved top20_grid.png")


# -----------------------------
# 3️⃣ Morgan Fingerprint Heatmap
# -----------------------------
generator = rdFingerprintGenerator.GetMorganGenerator(
    radius=2,
    fpSize=2048
)

fp = generator.GetFingerprint(mol)
arr = np.zeros((2048,))
DataStructs.ConvertToNumpyArray(fp, arr)

plt.figure(figsize=(12, 2))
plt.imshow(arr.reshape(1, -1), aspect="auto")
plt.yticks([])
plt.xlabel("Fingerprint Bit Index")
plt.title("Morgan Fingerprint")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fingerprint_heatmap.png")
plt.close()
print("✓ Saved fingerprint_heatmap.png")


# -----------------------------
# 4️⃣ Chemical Space PCA
# -----------------------------
all_acts = session.query(Bioactivity).all()

fps = []
values = []

for a in all_acts:
    m = Chem.MolFromSmiles(a.compound.smiles)
    if m and a.pchembl_value:
        f = generator.GetFingerprint(m)
        arr = np.zeros((2048,))
        DataStructs.ConvertToNumpyArray(f, arr)
        fps.append(arr)
        values.append(a.pchembl_value)

fps = np.array(fps)

pca = PCA(n_components=2)
coords = pca.fit_transform(fps)

plt.figure(figsize=(6, 6))
sc = plt.scatter(coords[:, 0], coords[:, 1], c=values)
plt.colorbar(sc, label="pChEMBL")
plt.title("Chemical Space (PCA)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/chemical_space.png")
plt.close()
print("✓ Saved chemical_space.png")


# -----------------------------
# 5️⃣ Training Curves (if exists)
# -----------------------------
metrics_path = "models/outputs/metrics.json"

if os.path.exists(metrics_path):
    with open(metrics_path) as f:
        metrics = json.load(f)

    if "train_loss" in metrics and "val_rmse" in metrics:
        plt.figure()
        plt.plot(metrics["train_loss"], label="Train Loss")
        plt.plot(metrics["val_rmse"], label="Val RMSE")
        plt.legend()
        plt.title("Training Curves")
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/training_curves_custom.png")
        plt.close()
        print("✓ Saved training_curves_custom.png")

    if "y_true" in metrics and "y_pred" in metrics:
        y_true = metrics["y_true"]
        y_pred = metrics["y_pred"]

        plt.figure(figsize=(6, 6))
        plt.scatter(y_true, y_pred)
        plt.plot(
            [min(y_true), max(y_true)],
            [min(y_true), max(y_true)]
        )
        plt.xlabel("Actual pChEMBL")
        plt.ylabel("Predicted pChEMBL")
        plt.title("Parity Plot")
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/parity_plot_custom.png")
        plt.close()
        print("✓ Saved parity_plot_custom.png")

print("\n[Viz] All visualizations saved to:")
print(f"   {OUTPUT_DIR}")