from datetime import datetime


# --------------------------------------------------
# Event priority
# --------------------------------------------------

EVENT_PRIORITY = {
    "POSSIBLE_ACCIDENT": "HIGH",
    "WRONG_WAY": "HIGH",
    "TRIPLE_RIDING": "MEDIUM",
    "NO_HELMET": "MEDIUM"
}


# --------------------------------------------------
# Event messages
# --------------------------------------------------

EVENT_MESSAGES = {

    "NO_HELMET":
        "⚠️ CityEye Alert\n"
        "No Helmet violation detected.",

    "TRIPLE_RIDING":
        "🚨 CityEye Alert\n"
        "Triple Riding detected.",

    "WRONG_WAY":
        "🚨 CityEye Alert\n"
        "Wrong-Way Driving detected.",

    "POSSIBLE_ACCIDENT":
        "🔴 CityEye HIGH PRIORITY Alert\n"
        "Possible Accident detected."
}


# --------------------------------------------------
# Build notification
# --------------------------------------------------

def build_notification(event_type, event):

    priority = EVENT_PRIORITY.get(
        event_type,
        "MEDIUM"
    )

    message = EVENT_MESSAGES.get(
        event_type,
        "CityEye Alert: Event detected."
    )

    timestamp = event.get(
        "timestamp",
        "N/A"
    )

    frame = event.get(
        "frame",
        "N/A"
    )

    message += (
        f"\n\nPriority: {priority}"
        f"\nFrame: {frame}"
        f"\nTimestamp: {timestamp}"
        f"\nTime: {datetime.now().strftime('%H:%M:%S')}"
    )

    return message


# --------------------------------------------------
# Send WhatsApp Alert
# --------------------------------------------------

def send_whatsapp_alert(event_type, event):

    message = build_notification(
        event_type,
        event
    )

    print("\n================================")
    print("📱 WHATSAPP ALERT - TEST MODE")
    print("================================")

    print(message)

    print("\nEvent Data:")
    print(event)

    print("--------------------------------")


# --------------------------------------------------
# Test
# --------------------------------------------------

if __name__ == "__main__":

    test_event = {
        "frame": 500,
        "timestamp": 20.0
    }

    send_whatsapp_alert(
        "POSSIBLE_ACCIDENT",
        test_event
    )