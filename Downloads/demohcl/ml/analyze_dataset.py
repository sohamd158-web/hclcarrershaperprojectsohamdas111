import os
import kagglehub
import pandas as pd

print("\nDownloading dataset...")

path = kagglehub.dataset_download(
    "ziya07/power-system-faults-dataset"
)

csv_path = os.path.join(
    path,
    "fault_data.csv"
)

df = pd.read_csv(csv_path)

print("\n" + "=" * 60)
print("DATASET ANALYSIS")
print("=" * 60)

print("\nShape:")
print(df.shape)

print("\nTarget distribution:")
print(df["Fault Type"].value_counts())



numeric_columns = [
    "Voltage (V)",
    "Current (A)",
    "Power Load (MW)",
    "Temperature (°C)",
    "Wind Speed (km/h)",
    "Duration of Fault (hrs)",
    "Down time (hrs)"
]

print("\n" + "=" * 60)
print("AVERAGE NUMERIC VALUES BY FAULT TYPE")
print("=" * 60)

print(
    df.groupby("Fault Type")[numeric_columns]
    .mean()
    .round(2)
)



print("\n" + "=" * 60)
print("MINIMUM VALUES BY FAULT TYPE")
print("=" * 60)

print(
    df.groupby("Fault Type")[numeric_columns]
    .min()
    .round(2)
)

print("\n" + "=" * 60)
print("MAXIMUM VALUES BY FAULT TYPE")
print("=" * 60)

print(
    df.groupby("Fault Type")[numeric_columns]
    .max()
    .round(2)
)




categorical_columns = [
    "Weather Condition",
    "Maintenance Status",
    "Component Health"
]

for column in categorical_columns:

    print("\n" + "=" * 60)
    print(column.upper())
    print("=" * 60)

    print(
        pd.crosstab(
            df["Fault Type"],
            df[column]
        )
    )




print("\n" + "=" * 60)
print("NUMERIC CORRELATION")
print("=" * 60)

numeric_df = df[numeric_columns]

print(
    numeric_df.corr().round(2)
)


print("\nAnalysis completed.")