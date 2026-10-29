import json
import ssl
import time
import random
import sys
import os

import certifi
import paho.mqtt.client as mqtt



BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

sys.path.append(BASE_DIR)

from config import (
    MQTT_BROKER,
    MQTT_PORT,
    MQTT_USERNAME,
    MQTT_PASSWORD,
    MQTT_TOPIC
)




DEVICE_ID = "SIMULATED_GRID_001"

SIMULATION_INTERVAL = 2



def generate_sensor_data():

   
    voltage_magnitude = round(
        random.uniform(210, 250),
        2
    )

    
    current_magnitude = round(
        random.uniform(5, 15),
        2
    )

    
    voltage_imbalance = round(
        random.uniform(0, 8),
        2
    )

    
    current_imbalance = round(
        random.uniform(0, 10),
        2
    )

    
    power = round(
        voltage_magnitude * current_magnitude,
        2
    )

    
    power_variation = round(
        random.uniform(-200, 200),
        2
    )

    data = {

        "device_id": DEVICE_ID,

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
            power_variation,

        "timestamp":
            time.strftime(
                "%Y-%m-%d %H:%M:%S"
            )
    }

    return data




def on_connect(
    client,
    userdata,
    flags,
    reason_code,
    properties=None
):

    if reason_code == 0:

        print("\n" + "=" * 65)
        print("        IoT ELECTRICAL MQTT PUBLISHER")
        print("=" * 65)

        print("SUCCESS: Connected to HiveMQ Cloud")
        print(f"Topic: {MQTT_TOPIC}")

        print("\nPublishing parameters:")
        print("• Voltage Magnitude")
        print("• Current Magnitude")
        print("• Voltage Imbalance")
        print("• Current Imbalance")
        print("• Power")
        print("• Power Variation")

        print("\nPress CTRL+C to stop.")
        print("=" * 65)

    else:

        print(
            f"ERROR: MQTT connection failed: "
            f"{reason_code}"
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


client.on_connect = on_connect




print("\nConnecting to HiveMQ Cloud...")

try:

    client.connect(
        MQTT_BROKER,
        MQTT_PORT,
        60
    )

except Exception as error:

    print("\nERROR: Could not connect to HiveMQ.")
    print(error)

    sys.exit(1)




client.loop_start()

time.sleep(2)




try:

    while True:

        data = generate_sensor_data()

        payload = json.dumps(data)

        result = client.publish(
            MQTT_TOPIC,
            payload,
            qos=1
        )

        if result.rc == mqtt.MQTT_ERR_SUCCESS:

            print("\n" + "-" * 65)

            print("DATA PUBLISHED SUCCESSFULLY")

            print("-" * 65)

            print(
                f"Voltage Magnitude : "
                f"{data['voltage_magnitude']} V"
            )

            print(
                f"Current Magnitude : "
                f"{data['current_magnitude']} A"
            )

            print(
                f"Voltage Imbalance : "
                f"{data['voltage_imbalance']} %"
            )

            print(
                f"Current Imbalance : "
                f"{data['current_imbalance']} %"
            )

            print(
                f"Power             : "
                f"{data['power']} W"
            )

            print(
                f"Power Variation   : "
                f"{data['power_variation']} W"
            )

        else:

            print(
                f"ERROR: Publish failed: "
                f"{result.rc}"
            )

        time.sleep(
            SIMULATION_INTERVAL
        )


except KeyboardInterrupt:

    print("\nStopping publisher...")


finally:

    client.loop_stop()

    client.disconnect()

    print(
        "Disconnected from HiveMQ Cloud."
    )