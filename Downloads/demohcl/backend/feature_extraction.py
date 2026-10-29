import pandas as pd
import numpy as np
from pathlib import Path




BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = BASE_DIR / "data" / "fault_data.csv"
OUTPUT_FILE = BASE_DIR / "data" / "processed_fault_data.csv"



def calculate_fault(row):

    if row["G"] == 1:
        return "Ground Fault"

    return "Normal"




def main():

    print("=" * 60)
    print("LOADING RAW DATASET")
    print("=" * 60)

    df = pd.read_csv(INPUT_FILE)

    print("Shape:", df.shape)
    print("Columns:", list(df.columns))


    
    required_columns = [
        "G",
        "C",
        "B",
        "A",
        "Ia",
        "Ib",
        "Ic",
        "Va",
        "Vb",
        "Vc"
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:

        raise ValueError(
            f"Missing columns: {missing}"
        )


    

    df["voltage_magnitude"] = np.sqrt(
        (
            df["Va"] ** 2
            + df["Vb"] ** 2
            + df["Vc"] ** 2
        ) / 3
    )


    
    df["current_magnitude"] = np.sqrt(
        (
            df["Ia"] ** 2
            + df["Ib"] ** 2
            + df["Ic"] ** 2
        ) / 3
    )


    

    voltage_abs = df[
        ["Va", "Vb", "Vc"]
    ].abs()

    voltage_mean = voltage_abs.mean(axis=1)

    voltage_deviation = (
        voltage_abs
        .sub(voltage_mean, axis=0)
        .abs()
        .max(axis=1)
    )

    df["voltage_imbalance"] = (
        voltage_deviation
        / (voltage_mean + 1e-6)
    ) * 100


   

    current_abs = df[
        ["Ia", "Ib", "Ic"]
    ].abs()

    current_mean = current_abs.mean(axis=1)

    current_deviation = (
        current_abs
        .sub(current_mean, axis=0)
        .abs()
        .max(axis=1)
    )

    df["current_imbalance"] = (
        current_deviation
        / (current_mean + 1e-6)
    ) * 100


    

    df["power"] = (
        df["voltage_magnitude"]
        * df["current_magnitude"]
    )


    

    df["power_variation"] = (
        df["power"]
        .diff()
        .fillna(0)
        .abs()
    )


    
    df["fault"] = df.apply(
        calculate_fault,
        axis=1
    )


    

    df.replace(
        [np.inf, -np.inf],
        np.nan,
        inplace=True
    )

    df.fillna(0, inplace=True)


    

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )


    

    print("\n" + "=" * 60)
    print("FEATURE EXTRACTION COMPLETE")
    print("=" * 60)


    print("\nExtracted Features:")

    features = [
        "voltage_magnitude",
        "current_magnitude",
        "voltage_imbalance",
        "current_imbalance",
        "power",
        "power_variation"
    ]

    for feature in features:

        print("✓", feature)


    print("\nFault Classes:")

    print(
        df["fault"].value_counts()
    )


    print("\nImbalance Statistics:")

    print(
        df[
            [
                "voltage_imbalance",
                "current_imbalance"
            ]
        ].describe()
    )


    print("\nProcessed Shape:")

    print(df.shape)


    print("\nSaved to:")

    print(OUTPUT_FILE)




if __name__ == "__main__":

    main()