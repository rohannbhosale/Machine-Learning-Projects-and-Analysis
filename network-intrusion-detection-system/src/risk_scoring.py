import numpy as np

# Severity weight per attack category — reflects real-world danger level, not just model confidence
ATTACK_SEVERITY = {
    "Normal": 0,
    "Analysis": 40,
    "Backdoor": 90,
    "DoS": 75,
    "Exploits": 85,
    "Fuzzers": 30,
    "Generic": 50,
    "Reconnaissance": 35,
    "Shellcode": 95,
    "Worms": 100,
}


def compute_risk_score(predicted_class, class_confidence, is_anomaly, anomaly_score):
    """
    Combines three signals into one 0-100 risk score.
    """
    # If the model says Normal AND nothing looks anomalous, risk should be near zero
    if predicted_class == "Normal" and not is_anomaly:
        return round(float(class_confidence * 5), 1)

    severity = ATTACK_SEVERITY.get(predicted_class, 50)
    anomaly_component = np.clip((-anomaly_score + 0.5) / 1.0, 0, 1)

    risk = (
        0.7 * severity * class_confidence +   # danger of the attack, weighted by how sure we are
        0.3 * (anomaly_component * 100)       # independent "looks unusual" signal
    )

    if is_anomaly and predicted_class != "Normal":
        risk = min(100, risk * 1.15)

    # Even if predicted Normal, being flagged anomalous should still raise some concern
    if predicted_class == "Normal" and is_anomaly:
        risk = max(risk, 25)

    return round(float(np.clip(risk, 0, 100)), 1)


if __name__ == "__main__":
    # Quick manual sanity checks
    print(compute_risk_score("Normal", 0.95, False, 0.1))       # should be low
    print(compute_risk_score("Worms", 0.9, True, -0.3))         # should be very high
    print(compute_risk_score("Fuzzers", 0.4, False, 0.05))      # should be low-moderate