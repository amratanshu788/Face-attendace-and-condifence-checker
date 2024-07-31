import cv2
import face_recognition
import pandas as pd
from datetime import datetime
import os

# Directory containing known face images
known_faces_dir = 'D:/Temporary'

# Directory to save captured images
captured_images_dir = 'D:/CapturedImages'

# Log file for errors
error_log_file = 'error_log.txt'

# Function to log errors


def log_error(message):
    with open(error_log_file, 'a') as f:
        f.write(f"{datetime.now()}: {message}\n")

# Load known faces


def load_known_faces(directory):
    known_faces = []
    known_names = []
    for filename in os.listdir(directory):
        if filename.lower().endswith(('.jpg', '.png')):
            try:
                image = face_recognition.load_image_file(
                    os.path.join(directory, filename))
                encoding = face_recognition.face_encodings(image)[0]
                known_faces.append(encoding)
                known_names.append(os.path.splitext(filename)[0])
            except Exception as e:
                log_error(f"Error processing file {filename}: {e}")
    return known_faces, known_names


known_faces, known_names = load_known_faces(known_faces_dir)

# Function to capture image


def capture_image():
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("Error: Could not open webcam.")
        return None

    print("Press Space to capture an image.")
    while True:
        ret, frame = cam.read()
        if not ret:
            print("Failed to grab frame")
            break

        cv2.imshow('Press Space to capture', frame)
        if cv2.waitKey(1) & 0xFF == ord(' '):
            # Save the captured image
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            image_path = os.path.join(
                captured_images_dir, f"captured_{timestamp}.jpg")
            cv2.imwrite(image_path, frame)
            print(f"Image saved at {image_path}")
            break

    cam.release()
    cv2.destroyAllWindows()
    return frame if ret else None

# Function to recognize face with confidence


def recognize_face(captured_image):
    face_encodings = face_recognition.face_encodings(captured_image)
    if len(face_encodings) == 0:
        print("No faces detected in the image.")
        return None, None

    captured_encoding = face_encodings[0]
    matches = face_recognition.compare_faces(known_faces, captured_encoding)
    face_distances = face_recognition.face_distance(
        known_faces, captured_encoding)

    if True in matches:
        first_match_index = matches.index(True)
        # Confidence in percentage
        confidence = round((1 - face_distances[first_match_index]) * 100, 2)
        return known_names[first_match_index], confidence
    return None, None

# Function to mark attendance


def mark_attendance(student_name, file='attendance.xlsx'):
    now = datetime.now()
    current_date = now.strftime("%Y-%m-%d")
    current_time = now.strftime("%H:%M:%S")

    try:
        df = pd.read_excel(file)
    except FileNotFoundError:
        df = pd.DataFrame(columns=["Name", "Date", "Time"])
    except Exception as e:
        log_error(f"Error reading the Excel file: {e}")
        return

    new_record_df = pd.DataFrame({"Name": [student_name], "Date": [
                                 current_date], "Time": [current_time]})
    df = pd.concat([df, new_record_df], ignore_index=True)

    try:
        df.to_excel(file, index=False)
    except Exception as e:
        log_error(f"Error saving the Excel file: {e}")

# Main execution


def main():
    image = capture_image()
    if image is None:
        return

    student_name, confidence = recognize_face(image)
    if student_name is None:
        print("Student not recognized!")
        return

    mark_attendance(student_name)
    if confidence:
        print(f"Attendance marked for {
              student_name} with confidence {confidence}%")
    else:
        print(f"Attendance marked for {student_name}")


if __name__ == "__main__":
    main()
