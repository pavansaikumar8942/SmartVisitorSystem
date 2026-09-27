
import csv
import io
import os
import sqlite3
from datetime import datetime

import cv2
from flask import Flask, Response, jsonify, render_template, request, send_file

app = Flask(__name__)

# --------------------------------------------------
# PATHS
# --------------------------------------------------

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATABASE = os.path.join(BASE_DIR, "visitors.db")
IMAGE_FOLDER = os.path.join(BASE_DIR, "static", "visitor_images")

os.makedirs(IMAGE_FOLDER, exist_ok=True)


# --------------------------------------------------
# DATABASE
# --------------------------------------------------

def get_db_connection():
    connection = sqlite3.connect(DATABASE)
    connection.row_factory = sqlite3.Row
    return connection


def init_db():
    connection = get_db_connection()

    connection.execute("""
        CREATE TABLE IF NOT EXISTS visitors (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image TEXT NOT NULL,
            entry_time TEXT NOT NULL
        )
    """)

    connection.commit()
    connection.close()


init_db()


# --------------------------------------------------
# FACE DETECTOR
# --------------------------------------------------

CASCADE_PATH = (
    cv2.data.haarcascades
    + "haarcascade_frontalface_default.xml"
)

face_detector = cv2.CascadeClassifier(CASCADE_PATH)

if face_detector.empty():
    raise RuntimeError("Could not load the face detector.")

camera = None

# Prevent repeated entries while a face stays visible
visitor_saved = False


def get_camera():
    global camera

    if camera is None or not camera.isOpened():
        camera = cv2.VideoCapture(0)

    return camera


# --------------------------------------------------
# SAVE VISITOR
# --------------------------------------------------

def save_visitor(frame):
    now = datetime.now()

    filename = now.strftime("visitor_%Y%m%d_%H%M%S_%f.jpg")
    image_path = os.path.join(IMAGE_FOLDER, filename)

    if not cv2.imwrite(image_path, frame):
        print("Could not save visitor image.")
        return False

    entry_time = now.strftime("%d-%m-%Y %I:%M:%S %p")

    connection = get_db_connection()

    connection.execute(
        "INSERT INTO visitors (image, entry_time) VALUES (?, ?)",
        (filename, entry_time),
    )

    connection.commit()
    connection.close()

    print("Visitor recorded:", entry_time)
    return True


# --------------------------------------------------
# CAMERA FRAME GENERATOR
# --------------------------------------------------

def generate_frames():
    global visitor_saved

    video_camera = get_camera()

    while True:
        success, frame = video_camera.read()

        if not success:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=3,
            minSize=(40, 40),
        )

        face_detected_now = len(faces) > 0

        # Draw a rectangle around each detected face
        for (x, y, width, height) in faces:
            cv2.rectangle(
                frame,
                (x, y),
                (x + width, y + height),
                (0, 255, 0),
                2,
            )

        if face_detected_now:
            if not visitor_saved:
                if save_visitor(frame):
                    visitor_saved = True

            cv2.putText(
                frame,
                "Visitor Already Recorded",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

        else:
            # Ready to record the next visitor
            visitor_saved = False

        success, buffer = cv2.imencode(".jpg", frame)

        if not success:
            continue

        frame_bytes = buffer.tobytes()

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + frame_bytes
            + b"\r\n"
        )


# --------------------------------------------------
# HOME PAGE
# --------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


# --------------------------------------------------
# LIVE CAMERA
# --------------------------------------------------

@app.route("/video")
def video():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


# --------------------------------------------------
# GET VISITOR RECORDS
# --------------------------------------------------

@app.route("/records")
def records():
    search_id = request.args.get("id", "").strip()
    selected_date = request.args.get("date", "").strip()

    query = """
        SELECT id, image, entry_time
        FROM visitors
        WHERE 1=1
    """

    parameters = []

    if search_id:
        if not search_id.isdigit():
            return jsonify([])

        query += " AND id = ?"
        parameters.append(int(search_id))

    if selected_date:
        try:
            formatted_date = datetime.strptime(
                selected_date, "%Y-%m-%d"
            ).strftime("%d-%m-%Y")
        except ValueError:
            return jsonify([])

        query += " AND substr(entry_time, 1, 10) = ?"
        parameters.append(formatted_date)

    query += " ORDER BY id DESC"

    connection = get_db_connection()
    rows = connection.execute(query, parameters).fetchall()
    connection.close()

    visitor_list = []

    for row in rows:
        visitor_list.append({
            "id": row["id"],
            "image": row["image"],
            "entry_time": row["entry_time"],
            "image_url": f"/static/visitor_images/{row['image']}",
        })

    return jsonify(visitor_list)


# --------------------------------------------------
# DELETE VISITOR AND RENUMBER IDs
# --------------------------------------------------

@app.route("/delete/<int:visitor_id>", methods=["DELETE"])
def delete_visitor(visitor_id):
    connection = get_db_connection()

    try:
        connection.execute("BEGIN IMMEDIATE")

        visitor = connection.execute(
            "SELECT image FROM visitors WHERE id = ?",
            (visitor_id,),
        ).fetchone()

        if visitor is None:
            connection.rollback()

            return jsonify({
                "success": False,
                "message": "Visitor not found.",
            }), 404

        # Delete selected visitor
        connection.execute(
            "DELETE FROM visitors WHERE id = ?",
            (visitor_id,),
        )

        # Temporarily change remaining IDs to negative values
        connection.execute("UPDATE visitors SET id = -id")

        # Restore IDs as 1, 2, 3...
        remaining_rows = connection.execute(
            "SELECT id FROM visitors ORDER BY id DESC"
        ).fetchall()

        for new_id, row in enumerate(remaining_rows, start=1):
            connection.execute(
                "UPDATE visitors SET id = ? WHERE id = ?",
                (new_id, row["id"]),
            )

        # Reset AUTOINCREMENT sequence
        count = len(remaining_rows)

        sequence_row = connection.execute(
            "SELECT name FROM sqlite_sequence WHERE name = 'visitors'"
        ).fetchone()

        if sequence_row:
            connection.execute(
                "UPDATE sqlite_sequence SET seq = ? WHERE name = 'visitors'",
                (count,),
            )
        else:
            connection.execute(
                "INSERT INTO sqlite_sequence (name, seq) VALUES ('visitors', ?)",
                (count,),
            )

        connection.commit()

        # Delete the image file
        image_path = os.path.join(IMAGE_FOLDER, visitor["image"])

        if os.path.exists(image_path):
            os.remove(image_path)

        return jsonify({
            "success": True,
            "message": "Visitor deleted and IDs renumbered.",
        })

    except Exception as error:
        connection.rollback()

        return jsonify({
            "success": False,
            "message": str(error),
        }), 500

    finally:
        connection.close()


# --------------------------------------------------
# EXPORT CSV
# --------------------------------------------------

@app.route("/export")
def export_csv():
    connection = get_db_connection()

    rows = connection.execute(
        "SELECT id, image, entry_time FROM visitors ORDER BY id ASC"
    ).fetchall()

    connection.close()

    output = io.StringIO()
    writer = csv.writer(output)

    writer.writerow([
        "Visitor ID",
        "Image Filename",
        "Entry Date and Time",
    ])

    for row in rows:
        writer.writerow([
            row["id"],
            row["image"],
            row["entry_time"],
        ])

    memory_file = io.BytesIO(output.getvalue().encode("utf-8-sig"))
    memory_file.seek(0)

    return send_file(
        memory_file,
        mimetype="text/csv",
        as_attachment=True,
        download_name="visitor_records.csv",
    )


# --------------------------------------------------
# RUN APPLICATION
# --------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True)