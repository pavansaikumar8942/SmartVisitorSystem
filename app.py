import os
import cv2
import csv
import base64
import sqlite3
import uuid
import io

from datetime import datetime
from flask import (
    Flask,
    render_template,
    request,
    jsonify,
    send_file,
    send_from_directory
)

app = Flask(__name__)

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

# Vercel allows temporary file storage in /tmp.
# Data stored here is not permanent.
BASE_DIR = "/tmp"

DATABASE = os.path.join(BASE_DIR, "visitors.db")
IMAGE_FOLDER = os.path.join(BASE_DIR, "visitor_images")

os.makedirs(IMAGE_FOLDER, exist_ok=True)

# OpenCV Haar Cascade face detector
CASCADE_PATH = (
    cv2.data.haarcascades
    + "haarcascade_frontalface_default.xml"
)

face_detector = cv2.CascadeClassifier(CASCADE_PATH)


# --------------------------------------------------
# DATABASE
# --------------------------------------------------

def get_db_connection():
    connection = sqlite3.connect(DATABASE)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_database():
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


initialize_database()


# --------------------------------------------------
# HOME PAGE
# --------------------------------------------------

@app.route("/")
def home():
    return render_template("index.html")


# --------------------------------------------------
# FACE DETECTION AND VISITOR CAPTURE
# --------------------------------------------------

@app.route("/api/detect", methods=["POST"])
def detect_face():
    try:
        data = request.get_json(silent=True)

        if not data or "image" not in data:
            return jsonify({
                "success": False,
                "message": "No image was provided."
            }), 400

        image_data = data["image"]

        # Remove the data URL prefix
        if "," in image_data:
            image_data = image_data.split(",", 1)[1]

        # Decode Base64 image
        image_bytes = base64.b64decode(image_data)

        image_array = (
            cv2.imdecode(
                __import__("numpy").frombuffer(
                    image_bytes,
                    dtype=__import__("numpy").uint8
                ),
                cv2.IMREAD_COLOR
            )
        )

        if image_array is None:
            return jsonify({
                "success": False,
                "message": "Invalid image."
            }), 400

        # Convert image to grayscale
        gray = cv2.cvtColor(
            image_array,
            cv2.COLOR_BGR2GRAY
        )

        # Detect faces
        faces = face_detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(50, 50)
        )

        if len(faces) == 0:
            return jsonify({
                "success": True,
                "face_detected": False,
                "message": "No face detected."
            })

        # Draw rectangles around detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(
                image_array,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

        # Create unique image filename
        image_name = f"{uuid.uuid4().hex}.jpg"
        image_path = os.path.join(
            IMAGE_FOLDER,
            image_name
        )

        # Save captured image
        cv2.imwrite(image_path, image_array)

        # Record entry time
        entry_time = datetime.now().strftime(
            "%d-%m-%Y %I:%M:%S %p"
        )

        # Save record to database
        connection = get_db_connection()

        cursor = connection.execute(
            """
            INSERT INTO visitors (image, entry_time)
            VALUES (?, ?)
            """,
            (image_name, entry_time)
        )

        visitor_id = cursor.lastrowid

        connection.commit()
        connection.close()

        return jsonify({
            "success": True,
            "face_detected": True,
            "visitor_id": visitor_id,
            "entry_time": entry_time,
            "image_url": f"/visitor-image/{image_name}",
            "message": "Visitor entry recorded."
        })

    except Exception as error:
        print("Face detection error:", error)

        return jsonify({
            "success": False,
            "message": "Unable to process the image."
        }), 500


# --------------------------------------------------
# DISPLAY SAVED VISITOR IMAGE
# --------------------------------------------------

@app.route("/visitor-image/<filename>")
def visitor_image(filename):
    return send_from_directory(
        IMAGE_FOLDER,
        filename
    )


# --------------------------------------------------
# GET VISITOR RECORDS
# --------------------------------------------------

@app.route("/api/records")
def get_records():
    connection = get_db_connection()

    records = connection.execute(
        """
        SELECT id, image, entry_time
        FROM visitors
        ORDER BY id DESC
        """
    ).fetchall()

    connection.close()

    visitor_list = []

    for record in records:
        visitor_list.append({
            "id": record["id"],
            "image": record["image"],
            "image_url": (
                f"/visitor-image/{record['image']}"
            ),
            "entry_time": record["entry_time"]
        })

    return jsonify(visitor_list)


# --------------------------------------------------
# RECORDS PAGE
# --------------------------------------------------

@app.route("/records")
def records():
    connection = get_db_connection()

    visitor_records = connection.execute(
        """
        SELECT id, image, entry_time
        FROM visitors
        ORDER BY id DESC
        """
    ).fetchall()

    connection.close()

    # If your project has a records.html template,
    # render it. Otherwise return the records as JSON.
    try:
        return render_template(
            "records.html",
            visitors=visitor_records
        )
    except Exception:
        return jsonify([
            {
                "id": record["id"],
                "image": record["image"],
                "entry_time": record["entry_time"]
            }
            for record in visitor_records
        ])


# --------------------------------------------------
# DELETE VISITOR RECORD
# --------------------------------------------------

@app.route("/delete/<int:visitor_id>", methods=["POST", "DELETE"])
def delete_visitor(visitor_id):
    connection = get_db_connection()

    record = connection.execute(
        "SELECT image FROM visitors WHERE id = ?",
        (visitor_id,)
    ).fetchone()

    if record is None:
        connection.close()

        return jsonify({
            "success": False,
            "message": "Visitor record not found."
        }), 404

    image_path = os.path.join(
        IMAGE_FOLDER,
        record["image"]
    )

    connection.execute(
        "DELETE FROM visitors WHERE id = ?",
        (visitor_id,)
    )

    connection.commit()
    connection.close()

    # Delete saved image if it exists
    if os.path.exists(image_path):
        os.remove(image_path)

    return jsonify({
        "success": True,
        "message": "Visitor record deleted."
    })


# --------------------------------------------------
# EXPORT VISITOR RECORDS TO CSV
# --------------------------------------------------

@app.route("/export")
def export_records():
    connection = get_db_connection()

    records = connection.execute(
        """
        SELECT id, image, entry_time
        FROM visitors
        ORDER BY id DESC
        """
    ).fetchall()

    connection.close()

    output = io.StringIO()
    writer = csv.writer(output)

    writer.writerow([
        "Visitor ID",
        "Image",
        "Entry Time"
    ])

    for record in records:
        writer.writerow([
            record["id"],
            record["image"],
            record["entry_time"]
        ])

    output.seek(0)

    csv_bytes = io.BytesIO(
        output.getvalue().encode("utf-8")
    )

    return send_file(
        csv_bytes,
        mimetype="text/csv",
        as_attachment=True,
        download_name="visitor_records.csv"
    )


# --------------------------------------------------
# OLD VIDEO ROUTE
# --------------------------------------------------

@app.route("/video")
def video():
    return jsonify({
        "message": (
            "The old server webcam stream is not used "
            "in the Vercel version. Use browser camera "
            "capture and send frames to /api/detect."
        )
    }), 410


# --------------------------------------------------
# LOCAL DEVELOPMENT
# --------------------------------------------------

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )