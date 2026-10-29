
VOLTAGE_MIN = 210
VOLTAGE_MAX = 250

CURRENT_MAX = 12

VOLTAGE_IMBALANCE_MAX = 5

CURRENT_IMBALANCE_MAX = 6

POWER_MAX = 3000

POWER_VARIATION_MAX = 150




def diagnose(data):

    faults = []


    

    voltage = float(
        data.get(
            "voltage_magnitude",
            0
        )
    )

    current = float(
        data.get(
            "current_magnitude",
            0
        )
    )

    voltage_imbalance = float(
        data.get(
            "voltage_imbalance",
            0
        )
    )

    current_imbalance = float(
        data.get(
            "current_imbalance",
            0
        )
    )

    power = float(
        data.get(
            "power",
            0
        )
    )

    power_variation = float(
        data.get(
            "power_variation",
            0
        )
    )


    

    if voltage > VOLTAGE_MAX:

        faults.append({

            "fault": "Overvoltage",

            "severity": "HIGH",

            "evidence":
                f"Voltage magnitude "
                f"{voltage:.2f} V exceeds "
                f"{VOLTAGE_MAX} V."
        })


    

    if voltage < VOLTAGE_MIN:

        faults.append({

            "fault": "Undervoltage",

            "severity": "MEDIUM",

            "evidence":
                f"Voltage magnitude "
                f"{voltage:.2f} V is below "
                f"{VOLTAGE_MIN} V."
        })


    
    if current > CURRENT_MAX:

        faults.append({

            "fault": "Overcurrent",

            "severity": "HIGH",

            "evidence":
                f"Current magnitude "
                f"{current:.2f} A exceeds "
                f"{CURRENT_MAX} A."
        })


   
    if voltage_imbalance > VOLTAGE_IMBALANCE_MAX:

        faults.append({

            "fault": "Voltage Imbalance",

            "severity": "HIGH",

            "evidence":
                f"Voltage imbalance "
                f"{voltage_imbalance:.2f}% exceeds "
                f"{VOLTAGE_IMBALANCE_MAX}%."
        })


    
    if current_imbalance > CURRENT_IMBALANCE_MAX:

        faults.append({

            "fault": "Current Imbalance",

            "severity": "HIGH",

            "evidence":
                f"Current imbalance "
                f"{current_imbalance:.2f}% exceeds "
                f"{CURRENT_IMBALANCE_MAX}%."
        })


    

    if power > POWER_MAX:

        faults.append({

            "fault": "Excessive Power",

            "severity": "MEDIUM",

            "evidence":
                f"Power "
                f"{power:.2f} W exceeds "
                f"{POWER_MAX} W."
        })


    
    if abs(power_variation) > POWER_VARIATION_MAX:

        faults.append({

            "fault":
                "Abnormal Power Variation",

            "severity":
                "MEDIUM",

            "evidence":
                f"Power variation "
                f"{power_variation:.2f} W exceeds "
                f"±{POWER_VARIATION_MAX} W."
        })


   

    return faults



def get_diagnosis_result(data):

    faults = diagnose(data)


    if not faults:

        return {

            "status":
                "NORMAL",

            "fault":
                "None",

            "severity":
                "LOW",

            "evidence":
                "All electrical parameters "
                "are within normal limits."
        }


    

    fault_names = "; ".join(
        fault["fault"]
        for fault in faults
    )

    severity_values = "; ".join(
        fault["severity"]
        for fault in faults
    )

    evidence_values = "; ".join(
        fault["evidence"]
        for fault in faults
    )


    return {

        "status":
            "FAULT DETECTED",

        "fault":
            fault_names,

        "severity":
            severity_values,

        "evidence":
            evidence_values
    }




if __name__ == "__main__":

    test_data = {

        "voltage_magnitude": 260,

        "current_magnitude": 14,

        "voltage_imbalance": 7,

        "current_imbalance": 8,

        "power": 3640,

        "power_variation": 190
    }


    result = get_diagnosis_result(
        test_data
    )


    print("\n" + "=" * 60)

    print(
        "        DIAGNOSIS ENGINE TEST"
    )

    print("=" * 60)

    print(
        f"Status   : {result['status']}"
    )

    print(
        f"Fault    : {result['fault']}"
    )

    print(
        f"Severity : {result['severity']}"
    )

    print(
        f"Evidence : {result['evidence']}"
    )

    print("=" * 60)