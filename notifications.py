import smtplib
from email.mime.text import MIMEText
import threading

# CONFIG (You would typically put this in a separate config file)
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
SENDER_EMAIL = "your_email@gmail.com"  # <--- CHANGE THIS
SENDER_PASSWORD = "your_app_password"  # <--- CHANGE THIS (Google App Password)

def send_attendance_email(recipient_email, student_name, time, status):
    def _send():
        try:
            subject = f"Attendance Alert: {student_name}"
            body = f"Hello,\n\n{student_name} has arrived at school.\nTime: {time}\nStatus: {status}\n\nRegards,\nSchool Admin"
            
            msg = MIMEText(body)
            msg['Subject'] = subject
            msg['From'] = SENDER_EMAIL
            msg['To'] = recipient_email

            with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
                server.starttls()
                server.login(SENDER_EMAIL, SENDER_PASSWORD)
                server.send_message(msg)
            print(f"Email sent to {recipient_email}")
        except Exception as e:
            print(f"Failed to send email: {e}")

    # Run in background thread so GUI doesn't freeze
    threading.Thread(target=_send, daemon=True).start()