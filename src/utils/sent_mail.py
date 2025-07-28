import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import time


def send_mail():
    sender_email = "vanlinh10042003@gmail.com"
    receiver_email = "vanlinh10042003@gmail.com"
    password = "aumk zwog upnn jglw"

    # Lấy thời gian hiện tại
    current_time = time.strftime("%Y-%m-%d %H:%M", time.localtime())

    # Tạo email
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = receiver_email
    msg['Subject'] = f"Báo động: Phát hiện BẤT THƯỜNG lúc {current_time}"

    body = f'''Xin chào,

Đây là một email báo động từ hệ thống camera giám sát của bạn.

Thời gian: {current_time}
Địa điểm: Trong nhà

Nội dung: Camera đã phát hiện một HÀNH ĐỘNG bất thường. Vui lòng kiểm tra ngay lập tức để đảm bảo an toàn.

Hãy hành động kịp thời và cẩn thận.

Trân trọng,
Hệ thống Giám sát '''
    msg.attach(MIMEText(body, 'plain'))

    try:
        # Kết nối tới máy chủ SMTP của Gmail
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, password)

        # Gửi email
        text = msg.as_string()
        server.sendmail(sender_email, receiver_email, text)
        server.quit()
    except Exception as e:
        print(f"Không thể gửi email. Lỗi: {e}")
