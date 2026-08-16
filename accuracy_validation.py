from pathlib import Path

CLIP_DIR = Path("output/event_clips")

clips = sorted(CLIP_DIR.glob("*.mp4"))

if not clips:
    print("No event clips found.")
    exit()

print("\n================================")
print("CITYEYE ACCURACY VALIDATION")
print("================================")
print(f"Total AI detected events: {len(clips)}")
print()
print("For each clip:")
print("Y = Event is actually present")
print("N = False detection")
print()

correct = 0
incorrect = 0

for i, clip in enumerate(clips, 1):
    print(f"\n{i}/{len(clips)} : {clip.name}")

    while True:
        answer = input("Correct event? (Y/N): ").strip().upper()

        if answer == "Y":
            correct += 1
            break
        elif answer == "N":
            incorrect += 1
            break
        else:
            print("Please enter only Y or N.")

total = correct + incorrect

precision = (correct / total) * 100 if total else 0

print("\n================================")
print("CITYEYE VALIDATION RESULT")
print("================================")
print(f"Total AI detections : {total}")
print(f"Correct detections  : {correct}")
print(f"False detections    : {incorrect}")
print(f"Detection Precision : {precision:.2f}%")
print("================================")