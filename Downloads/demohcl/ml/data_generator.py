import sys
import json
import time
import ssl
import certifi
import random

from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import joblib
import paho.mqtt.client as mqtt




BASE_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = (
    BASE_DIR
    / "data"
    / "processed_fault_data.csv"
)

MODEL_FILE = (
    BASE_DIR
    / "model"
    / "fault_model.pkl"
)



sys.path.insert(
    0,
    str(BASE_DIR)
)

from config import (
    MQTT_BROKER,
    MQTT_PORT,
    MQTT_USERNAME,
    MQTT_PASSWORD,
    MQTT_TOPIC,
    DEVICE_ID,
    PUBLISH_INTERVAL
)



FEATURES = [
    "voltage_magnitude",
    "current_magnitude",
    "voltage_imbalance",
    "current_imbalance",
    "power",
    "power_variation"
]




if not DATA_FILE.exists():

    print("ERROR: Processed dataset not found.")

    print(
        "Expected:"
    )

    print(DATA_FILE)

    print(
        "\nRun:"
    )

    print(
        "python3 backend/feature_extraction.py"
    )

    sys.exit(1)


if not MODEL_FILE.exists():

    print("ERROR: Trained ML model not found.")

    print(
        "Expected:"
    )

    print(MODEL_FILE)

    print(
        "\nRun:"
    )

    print(
        "python3 ml/train_model.py"
    )

    sys.exit(1)




print("=" * 60)

print(
    "ML SENSOR DATA GENERATOR"
)

print("=" * 60)



print(
    "\nLoading processed dataset..."
)

df = pd.read_csv(
    DATA_FILE
)

print(
    "Rows:",
    len(df)
)



missing_features = [
    feature
    for feature in FEATURES
    if feature not in df.columns
]

if missing_features:

    print(
        "ERROR: Missing features:"
    )

    print(
        missing_features
    )

    sys.exit(1)




print(
    "\nLoading trained ML model..."
)

model_bundle = joblib.load(
    MODEL_FILE
)

fault_model = model_bundle[
    "fault_model"
]

print(
    "Random Forest loaded successfully."
)



normal_df = df[
    df["fault"] == "Normal"
].copy()


fault_df = df[
    df["fault"] == "Ground Fault"
].copy()


print(
    "\nLearned patterns:"
)

print(
    "Normal:",
    len(normal_df)
)

print(
    "Ground Fault:",
    len(fault_df)
)




def generate_sample():

    

    intended_class = random.choices(
        [
            "Normal",
            "Ground Fault"
        ],
        weights=[
            70,
            30
        ],
        k=1
    )[0]


    
    if intended_class == "Normal":

        source_row = normal_df.sample(
            n=1
        ).iloc[0]

    else:

        source_row = fault_df.sample(
            n=1
        ).iloc[0]


    

    generated = {}

    for feature in FEATURES:

        value = float(
            source_row[feature]
        )

       

        variation = np.random.normal(
            loc=0.0,
            scale=0.03
        )

        value = value * (
            1 + variation
        )


        

        if feature in [
            "voltage_magnitude",
            "current_magnitude",
            "power",
            "power_variation"
        ]:

            value = abs(value)


        generated[feature] = float(
            value
        )


    
    generated["power"] = (
        generated["voltage_magnitude"]
        *
        generated["current_magnitude"]
    )


    

    generated[
        "voltage_imbalance"
    ] = max(
        0.0,
        generated["voltage_imbalance"]
    )


    generated[
        "current_imbalance"
    ] = max(
        0.0,
        generated["current_imbalance"]
    )


    return (
        generated,
        intended_class
    )




client = mqtt.Client(
    mqtt.CallbackAPIVersion.VERSION2
)

client.username_pw_set(
    MQTT_USERNAME,
    MQTT_PASSWORD
)


client.tls_set(
    ca_certs=certifi.where(),
    tls_version=ssl.PROTOCOL_TLS_CLIENT
)



print(
    "\nConnecting to HiveMQ Cloud..."
)

try:

    client.connect(
        MQTT_BROKER,
        MQTT_PORT,
        60
    )

except Exception as error:

    print(
        "\nERROR: HiveMQ connection failed."
    )

    print(
        type(error).__name__,
        ":",
        error
    )

    sys.exit(1)


client.loop_start()


print(
    "\nConnected successfully."
)


print(
    "\n" + "=" * 60
)

print(
    "ML GENERATOR RUNNING"
)

print(
    "=" * 60
)

print(
    "MQTT Topic :",
    MQTT_TOPIC
)

print(
    "Device     :",
    DEVICE_ID
)

print(
    "Interval   :",
    PUBLISH_INTERVAL,
    "seconds"
)

print(
    "\nNew sensor data is being generated "
    "from learned dataset patterns."
)

print(
    "Press CTRL+C to stop.\n"
)




try:

    while True:

       

        sensor_data, intended_class = (
            generate_sample()
        )


       

        X = np.array(
            [
                [
                    sensor_data[feature]
                    for feature in FEATURES
                ]
            ],
            dtype=float
        )


        

        predicted_class = (
            fault_model.predict(X)[0]
        )


        probabilities = (
            fault_model.predict_proba(X)[0]
        )


        confidence = float(
            np.max(probabilities)
        )


        

        payload = {

            "timestamp":
                datetime.now(
                    timezone.utc
                ).isoformat(),

            "device_id":
                DEVICE_ID,

            "voltage_magnitude":
                sensor_data[
                    "voltage_magnitude"
                ],

            "current_magnitude":
                sensor_data[
                    "current_magnitude"
                ],

            "voltage_imbalance":
                sensor_data[
                    "voltage_imbalance"
                ],

            "current_imbalance":
                sensor_data[
                    "current_imbalance"
                ],

            "power":
                sensor_data[
                    "power"
                ],

            "power_variation":
                sensor_data[
                    "power_variation"
                ]
        }


       
        result = client.publish(
            MQTT_TOPIC,
            json.dumps(payload),
            qos=1
        )

        result.wait_for_publish()


        
        print(
            "=" * 60
        )

        print(
            "NEW ML-GENERATED SENSOR DATA"
        )

        print(
            "=" * 60
        )

        print(
            "Intended Pattern :",
            intended_class
        )

        print(
            "ML Prediction    :",
            predicted_class
        )

        print(
            "ML Confidence    :",
            f"{confidence * 100:.2f}%"
        )

        print(
            "Voltage          :",
            f"{sensor_data['voltage_magnitude']:.4f}"
        )

        print(
            "Current          :",
            f"{sensor_data['current_magnitude']:.4f}"
        )

        print(
            "Voltage Imbalance:",
            f"{sensor_data['voltage_imbalance']:.4f}%"
        )

        print(
            "Current Imbalance:",
            f"{sensor_data['current_imbalance']:.4f}%"
        )

        print(
            "Power            :",
            f"{sensor_data['power']:.4f}"
        )

        print(
            "Power Variation  :",
            f"{sensor_data['power_variation']:.4f}"
        )

        print(
            "\n✓ New data published to HiveMQ"
        )


        
        time.sleep(
            PUBLISH_INTERVAL
        )




except KeyboardInterrupt:

    print(
        "\n\nStopping ML data generator..."
    )


finally:

    client.loop_stop()

    client.disconnect()

    print(
        "ML data generator stopped."
    )