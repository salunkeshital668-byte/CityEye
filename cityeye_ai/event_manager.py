from pathlib import Path
import json

from notification_manager import send_whatsapp_alert


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"


def load_json(filename):
    path = DATA_DIR / filename

    if not path.exists():
        print(f"Warning: {filename} not found")
        return {}

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def group_no_helmet_events(events, frame_gap=30):

    if not events:
        return []

    events = sorted(
        events,
        key=lambda x: x.get("frame", 0)
    )

    groups = []
    current_group = [events[0]]

    for event in events[1:]:

        current_frame = event.get("frame", 0)
        previous_frame = current_group[-1].get("frame", 0)

        if current_frame - previous_frame <= frame_gap:
            current_group.append(event)
        else:
            groups.append(current_group)
            current_group = [event]

    groups.append(current_group)

    final_events = []

    for group in groups:

        confidence_values = [
            event.get("confidence", 0)
            for event in group
        ]

        final_events.append({
            "type": "NO_HELMET",
            "start_frame": group[0].get("frame"),
            "end_frame": group[-1].get("frame"),
            "detections": len(group),
            "confidence": round(
                max(confidence_values),
                3
            )
        })

    return final_events


def unique_by_track(events):

    unique = {}
    no_id_events = []

    for event in events:

        track_id = (
            event.get("track_id")
            or event.get("motorcycle_id")
        )

        if track_id is None:
            no_id_events.append(event)
            continue

        if track_id not in unique:
            unique[track_id] = event

    return list(unique.values()) + no_id_events


def unique_accidents(events):

    unique = {}

    for event in events:

        vehicle1 = event.get("vehicle_1", {})
        vehicle2 = event.get("vehicle_2", {})

        id1 = vehicle1.get("track_id")
        id2 = vehicle2.get("track_id")

        if id1 is None or id2 is None:
            continue

        pair = tuple(sorted([id1, id2]))

        if pair not in unique:
            unique[pair] = event

    return list(unique.values())


def send_event_notifications(
    helmet_events,
    triple_events,
    wrong_way_events,
    accident_events
):

    print("\n================================")
    print("CITYEYE NOTIFICATION SYSTEM")
    print("================================")

    # No Helmet
    for event in helmet_events:

        send_whatsapp_alert(
            "NO_HELMET",
            event
        )

    # Triple Riding
    for event in triple_events:

        send_whatsapp_alert(
            "TRIPLE_RIDING",
            event
        )

    # Wrong Way
    for event in wrong_way_events:

        send_whatsapp_alert(
            "WRONG_WAY",
            event
        )

    # Accident
    for event in accident_events:

        send_whatsapp_alert(
            "POSSIBLE_ACCIDENT",
            event
        )


def merge_events():

    helmet = load_json("helmet_events.json")
    triple = load_json("triple_riding_events.json")
    wrong_way = load_json("wrong_way_events.json")
    accident = load_json("accident_events.json")

    helmet_raw = helmet.get("events", [])
    triple_raw = triple.get("events", [])
    wrong_way_raw = wrong_way.get("events", [])
    accident_raw = accident.get("events", [])

    helmet_events = group_no_helmet_events(
        helmet_raw,
        frame_gap=30
    )

    triple_events = unique_by_track(
        triple_raw
    )

    wrong_way_events = unique_by_track(
        wrong_way_raw
    )

    accident_events = unique_accidents(
        accident_raw
    )

    final_result = {

        "project": "CityEye",

        "video": "input.mp4",

        "summary": {

            "raw_detections": {
                "no_helmet": len(helmet_raw),
                "triple_riding": len(triple_raw),
                "wrong_way": len(wrong_way_raw),
                "accident": len(accident_raw)
            },

            "unique_events": {
                "no_helmet": len(helmet_events),
                "triple_riding": len(triple_events),
                "wrong_way": len(wrong_way_events),
                "accident": len(accident_events)
            }
        },

        "events": {

            "no_helmet": helmet_events,

            "triple_riding": triple_events,

            "wrong_way": wrong_way_events,

            "accident": accident_events
        }
    }

    output_file = DATA_DIR / "final_events.json"

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            final_result,
            file,
            indent=4
        )

    print("\n================================")
    print("CITYEYE FINAL EVENT SUMMARY")
    print("================================")

    print("\nRAW DETECTIONS")
    print("--------------------------------")
    print(f"No Helmet      : {len(helmet_raw)}")
    print(f"Triple Riding  : {len(triple_raw)}")
    print(f"Wrong Way      : {len(wrong_way_raw)}")
    print(f"Accident       : {len(accident_raw)}")

    print("\nUNIQUE EVENTS")
    print("--------------------------------")
    print(f"No Helmet      : {len(helmet_events)}")
    print(f"Triple Riding  : {len(triple_events)}")
    print(f"Wrong Way      : {len(wrong_way_events)}")
    print(f"Accident       : {len(accident_events)}")

    total_unique = (
        len(helmet_events)
        + len(triple_events)
        + len(wrong_way_events)
        + len(accident_events)
    )

    print(f"\nTotal Unique Events : {total_unique}")

    print("--------------------------------")
    print(f"Saved: {output_file}")

    # Send TEST notifications
    send_event_notifications(
        helmet_events,
        triple_events,
        wrong_way_events,
        accident_events
    )


if __name__ == "__main__":
    merge_events()