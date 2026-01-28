import face_recognition as fr
import cv2
import numpy as np
import os

# ===============================
# TRAINING PHASE
# ===============================
train_path = "./train/"

known_names = []
known_encodings = []

print("🔹 Training faces...")

for file in os.listdir(train_path):

    # Allow only image files
    if not file.lower().endswith((".jpg", ".jpeg", ".png")):
        print(f"⏭ Skipping non-image file: {file}")
        continue

    file_path = os.path.join(train_path, file)

    try:
        img = fr.load_image_file(file_path)
        encodings = fr.face_encodings(img)

        if len(encodings) > 0:
            known_encodings.append(encodings[0])
            name = os.path.splitext(file)[0]
            known_names.append(name)
            print(f"✅ Loaded: {name}")
        else:
            print(f"⚠ No face found in: {file}")

    except Exception as e:
        print(f"❌ Error loading {file}: {e}")

print("\nTrained Faces:", known_names)

# ===============================
# TESTING PHASE (MULTIPLE IMAGES)
# ===============================
test_path = "./test/"

for test_file in os.listdir(test_path):

    if not test_file.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    img_path = os.path.join(test_path, test_file)
    image = cv2.imread(img_path)

    if image is None:
        print(f"❌ Cannot read image: {test_file}")
        continue

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    face_locations = fr.face_locations(rgb_image)
    face_encodings = fr.face_encodings(rgb_image, face_locations)

    print(f"\n🖼 Processing {test_file} | Faces found: {len(face_encodings)}")

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):

        matches = fr.compare_faces(known_encodings, face_encoding)
        face_distances = fr.face_distance(known_encodings, face_encoding)

        name = "Unknown"

        if len(face_distances) > 0:
            best_match = np.argmin(face_distances)
            if matches[best_match]:
                name = known_names[best_match]

        # Draw rectangle and name
        cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.rectangle(image, (left, bottom - 30), (right, bottom), (0, 0, 255), cv2.FILLED)
        cv2.putText(
            image,
            name,
            (left + 6, bottom - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

    # Save output image
    output_path = f"./output_{test_file}"
    cv2.imwrite(output_path, image)
    print(f"✅ Saved result as {output_path}")

print("\n🎉 Face Recognition Completed")
