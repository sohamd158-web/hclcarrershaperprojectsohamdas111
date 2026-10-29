import pandas as pd
import numpy as np



DATASET_PATH = "data/fault_data.csv"

df = pd.read_csv(DATASET_PATH)

print("=" * 70)
print("NEW ELECTRICAL FAULT DATASET")
print("=" * 70)

print("Shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())



df["voltage_magnitude"] = np.sqrt(
    (
        df["Va"] ** 2 +
        df["Vb"] ** 2 +
        df["Vc"] ** 2
    ) / 3
)



df["current_magnitude"] = np.sqrt(
    (
        df["Ia"] ** 2 +
        df["Ib"] ** 2 +
        df["Ic"] ** 2
    ) / 3
)



voltage_abs = df[["Va", "Vb", "Vc"]].abs()

voltage_mean = voltage_abs.mean(axis=1)

df["voltage_imbalance"] = np.where(
    voltage_mean != 0,
    (
        (voltage_abs.max(axis=1) - voltage_mean)
        / voltage_mean
    ) * 100,
    0
)



current_abs = df[["Ia", "Ib", "Ic"]].abs()

current_mean = current_abs.mean(axis=1)

df["current_imbalance"] = np.where(
    current_mean != 0,
    (
        (current_abs.max(axis=1) - current_mean)
        / current_mean
    ) * 100,
    0
)



df["power"] = (
    df["voltage_magnitude"] *
    df["current_magnitude"]
)



df["power_variation"] = (
    df["power"].diff().fillna(0)
)



def get_fault(row):

    G = row["G"]
    C = row["C"]
    B = row["B"]
    A = row["A"]

    if G == 0 and A == 0 and B == 0 and C == 0:
        return "Normal"

    if G == 1 and A == 1 and B == 0 and C == 0:
        return "Ground-A Fault"

    if G == 1 and B == 1 and A == 0 and C == 0:
        return "Ground-B Fault"

    if G == 1 and C == 1 and A == 0 and B == 0:
        return "Ground-C Fault"

    if A == 1 and B == 0 and C == 0:
        return "Phase-A Fault"

    if B == 1 and A == 0 and C == 0:
        return "Phase-B Fault"

    if C == 1 and A == 0 and B == 0:
        return "Phase-C Fault"

    if A == 1 and B == 1 and C == 0:
        return "A-B Fault"

    if A == 1 and C == 1 and B == 0:
        return "A-C Fault"

    if B == 1 and C == 1 and A == 0:
        return "B-C Fault"

    if A == 1 and B == 1 and C == 1:
        return "A-B-C Fault"

    return "Other Fault"


df["fault"] = df.apply(get_fault, axis=1)




features = [
    "voltage_magnitude",
    "current_magnitude",
    "voltage_imbalance",
    "current_imbalance",
    "power",
    "power_variation"
]




output_columns = features + ["fault"]

processed = df[output_columns]

processed.to_csv(
    "data/processed_fault_data.csv",
    index=False
)




print("\n" + "=" * 70)
print("FEATURE EXTRACTION COMPLETE")
print("=" * 70)

print("\nFinal features:")

for feature in features:
    print("✓", feature)

print("\nFault classes:")
print(processed["fault"].value_counts())

print("\nProcessed dataset shape:")
print(processed.shape)

print("\nProcessed dataset:")
print(processed.head(10))

print("\nSaved to:")
print("data/processed_fault_data.csv")