import sys
import json
import ssl
import certifi
import joblib
import numpy as np

from pathlib import Path

import paho.mqtt.client as mqtt




BASE_DIR = Path(__file__).resolve().parent.parent


sys.path.insert(0, str(BASE_DIR))


sys.path.insert(0, str(BASE_DIR / "backend"))




from config import (
    MQTT_BROKER,
    MQTT_PORT,
    MQTT_USERNAME,
    MQTT_PASSWORD,
    MQTT_TOPIC
)

from database import (
    initialize_database,
    save_raw_data,
    save_analysis
)




MODEL_FILE = (
    BASE_DIR /
    "model" /
    "fault_model.pkl"
)




print("=" * 60)
print("LOADING ML MODEL")
print("=" * 60)

if not MODEL_FILE.exists():
    print("ERROR: ML model not found.")
    print("Expected location:")
    print(MODEL_FILE)
    print("\nRun:")
    print("python3 backend/train_model.py")
    sys.exit(1)

model_bundle = joblib.load(
    MODEL_FILE
)

fault_model = model_bundle["fault_model"]

anomaly_model = model_bundle["anomaly_model"]

FEATURES = model_bundle["features"]

print("ML models loaded successfully.")




initialize_database()

print("SQLite database ready.")

print(
    "Database:",
    BASE_DIR / "data" / "iot_fault_diagnosis.db"
)



def calculate_severity(
    predicted_fault,
    confidence,
    anomaly
):

    if anomaly == "ANOMALY":
        return "HIGH"

    if predicted_fault != "Normal":

        if confidence >= 0.90:
            return "HIGH"

        return "MEDIUM"

    return "LOW"




def create_evidence(data):

    evidence = []

    if data["current_imbalance"] > 6:
        evidence.append(
            "High current imbalance"
        )

    if data["voltage_imbalance"] > 5:
        evidence.append(
            "High voltage imbalance"
        )

    if data["power_variation"] > 150:
        evidence.append(
            "High power variation"
        )

    if not evidence:

        evidence.append(
            "Sensor values within learned normal pattern"
        )

    return "; ".join(evidence)




def on_connect(
    client,
    userdata,
    flags,
    reason_code,
    properties=None
):

    print()

    if reason_code == 0:

        print("=" * 60)
        print("MQTT CONNECTED")
        print("=" * 60)

        print(
            "Broker:",
            MQTT_BROKER
        )

        print(
            "Topic:",
            MQTT_TOPIC
        )

        result = client.subscribe(
            MQTT_TOPIC,
            qos=1
        )

        print(
            "Subscription result:",
            result
        )

        print(
            "\nWaiting for sensor data..."
        )

    else:

        print(
            "MQTT connection failed."
        )

        print(
            "Reason code:",
            reason_code
        )




def on_disconnect(
    client,
    userdata,
    disconnect_flags,
    reason_code,
    properties=None
):

    print(
        "\nMQTT disconnected."
    )

    print(
        "Reason code:",
        reason_code
    )




def on_message(
    client,
    userdata,
    msg
):

    try:

       

        payload = msg.payload.decode(
            "utf-8"
        )

        data = json.loads(
            payload
        )

        print("\n" + "=" * 60)
        print("NEW SENSOR DATA")
        print("=" * 60)

        print(
            json.dumps(
                data,
                indent=2
            )
        )


        

        raw_data_id = save_raw_data(
            data
        )

        print(
            "\n✓ Raw data stored in SQLite"
        )

        print(
            "Raw Data ID:",
            raw_data_id
        )


        

        feature_values = []

        for feature in FEATURES:

            if feature not in data:

                raise ValueError(
                    f"Missing feature: {feature}"
                )

            feature_values.append(
                float(data[feature])
            )


        X = np.array(
            [feature_values],
            dtype=float
        )


        
        predicted_fault = (
            fault_model
            .predict(X)[0]
        )

        probabilities = (
            fault_model
            .predict_proba(X)[0]
        )

        confidence = float(
            np.max(probabilities)
        )


        

        anomaly_prediction = (
            anomaly_model
            .predict(X)[0]
        )

        anomaly_score = float(
            anomaly_model
            .decision_function(X)[0]
        )

        if anomaly_prediction == -1:

            anomaly = "ANOMALY"

        else:

            anomaly = "NORMAL"


        

        severity = calculate_severity(
            predicted_fault,
            confidence,
            anomaly
        )


        

        evidence = create_evidence(
            data
        )


        

        save_analysis(

            raw_data_id,

            predicted_fault,

            confidence,

            anomaly,

            anomaly_score,

            severity,

            evidence
        )


        

        print("\n" + "=" * 60)
        print("ML DIAGNOSIS RESULT")
        print("=" * 60)

        print(
            "Predicted Fault :",
            predicted_fault
        )

        print(
            "Confidence      :",
            f"{confidence * 100:.2f}%"
        )

        print(
            "Anomaly         :",
            anomaly
        )

        print(
            "Anomaly Score   :",
            f"{anomaly_score:.4f}"
        )

        print(
            "Severity        :",
            severity
        )

        print(
            "Evidence        :",
            evidence
        )

        print(
            "\n✓ Analysis saved in SQLite"
        )


    except json.JSONDecodeError:

        print(
            "\nERROR: Invalid JSON received."
        )

        print(
            "Payload:",
            msg.payload.decode(
                "utf-8",
                errors="replace"
            )
        )


    except Exception as error:

        print(
            "\nERROR processing MQTT message:"
        )

        print(
            type(error).__name__,
            ":",
            error
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



client.on_connect = on_connect

client.on_message = on_message

client.on_disconnect = on_disconnect




print("\n" + "=" * 60)
print("MQTT SUBSCRIBER")
print("=" * 60)

print(
    "Connecting to HiveMQ Cloud..."
)

try:

    client.connect(
        MQTT_BROKER,
        MQTT_PORT,
        60
    )

except Exception as error:

    print(
        "\nERROR: Could not connect to HiveMQ Cloud."
    )

    print(
        type(error).__name__,
        ":",
        error
    )

    print(
        "\nCheck these values in config.py:"
    )

    print(
        "MQTT_BROKER"
    )

    print(
        "MQTT_PORT = 8883"
    )

    print(
        "MQTT_USERNAME"
    )

    print(
        "MQTT_PASSWORD"
    )

    sys.exit(1)




print(
    "\nSubscriber is running..."
)

print(
    "Press CTRL+C to stop."
)

try:

    client.loop_forever()

except KeyboardInterrupt:

    print(
        "\nStopping subscriber..."
    )

    client.disconnect()

    print(
        "Subscriber stopped.")
    