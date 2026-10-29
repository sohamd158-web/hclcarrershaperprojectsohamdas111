import sys
import json
import time
import random
import ssl
import certifi

from pathlib import Path
from datetime import datetime, timezone



BASE_DIR = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(BASE_DIR))



import pandas as pd
import paho.mqtt.client as mqtt

from config import (
    MQTT_BROKER,
    MQTT_PORT,
    MQTT_USERNAME,
    MQTT_PASSWORD,
    MQTT_TOPIC,
    DEVICE_ID,
    PUBLISH_INTERVAL
)



DATA_FILE = (
    BASE_DIR /
    "data" /
    "processed_fault_data.csv"
)

if not DATA_FILE.exists():

    print("ERROR: Processed dataset not found.")

    print(
        "Expected:",
        DATA_FILE
    )

    print(
        "\nRun this first:"
    )

    print(
        "python3 backend/feature_extraction.py"
    )

    sys.exit(1)


print("=" * 60)
print("SENSOR SIMULATOR")
print("=" * 60)

print(
    "Loading:",
    DATA_FILE
)

df = pd.read_csv(
    DATA_FILE
)

print(
    "Loaded rows:",
    len(df)
)




def add_noise(value):

    value = float(value)

    noise_scale = max(
        abs(value) * 0.005,
        0.000001
    )

    return float(
        value +
        random.gauss(
            0,
            noise_scale
        )
    )




client = mqtt.Client(
    mqtt.CallbackAPIVersion.VERSION2
)

client.username_pw_set(
    MQTT_USERNAME,
    MQTT_PASSWORD
)




print(
    "\nUsing CA certificate:"
)

print(
    certifi.where()
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
    "Connected successfully."
)

print(
    "\nPublishing to:",
    MQTT_TOPIC
)

print(
    "Device:",
    DEVICE_ID
)

print(
    "\nPress CTRL+C to stop.\n"
)



try:

    while True:

        
        row = df.sample(
            n=1
        ).iloc[0]


        

        voltage_magnitude = add_noise(
            row["voltage_magnitude"]
        )

        current_magnitude = add_noise(
            row["current_magnitude"]
        )

        voltage_imbalance = add_noise(
            row["voltage_imbalance"]
        )

        current_imbalance = add_noise(
            row["current_imbalance"]
        )


        

        power = (
            voltage_magnitude *
            current_magnitude
        )


        

        power_variation = add_noise(
            row["power_variation"]
        )


        

        payload = {

            "timestamp":
                datetime.now(
                    timezone.utc
                ).isoformat(),

            "device_id":
                DEVICE_ID,

            "voltage_magnitude":
                voltage_magnitude,

            "current_magnitude":
                current_magnitude,

            "voltage_imbalance":
                voltage_imbalance,

            "current_imbalance":
                current_imbalance,

            "power":
                power,

            "power_variation":
                power_variation
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
            "DATA PUBLISHED"
        )

        print(
            "Timestamp:",
            payload["timestamp"]
        )

        print(
            "Device:",
            payload["device_id"]
        )

        print(
            "Voltage:",
            f"{payload['voltage_magnitude']:.4f}"
        )

        print(
            "Current:",
            f"{payload['current_magnitude']:.4f}"
        )

        print(
            "Voltage Imbalance:",
            f"{payload['voltage_imbalance']:.4f}"
        )

        print(
            "Current Imbalance:",
            f"{payload['current_imbalance']:.4f}"
        )

        print(
            "Power:",
            f"{payload['power']:.4f}"
        )

        print(
            "Power Variation:",
            f"{payload['power_variation']:.4f}"
        )


        time.sleep(
            PUBLISH_INTERVAL
        )


except KeyboardInterrupt:

    print(
        "\nStopping sensor simulator..."
    )


finally:

    client.loop_stop()

    client.disconnect()

    print(
        "Simulator stopped."
    )