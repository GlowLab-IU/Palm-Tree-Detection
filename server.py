# -*- coding: utf-8 -*-
"""
Server launcher for Palm Tree Counting API (Local CUDA-forced)
"""
import uvicorn
import os
import logging
from dotenv import load_dotenv

# Tải các biến môi trường từ file .env lên đầu tiên
load_dotenv()

# --- Yêu cầu sử dụng CUDA ---
# Đặt biến môi trường này TRƯỚC KHI import 'main'
# để đảm bảo model (ví dụ: PyTorch/SAHI) được tải lên GPU.
# "0" là chỉ số của GPU đầu tiên.
# Nếu không có GPU, việc này sẽ khiến ứng dụng bị lỗi khi tải model.
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# --- Cấu hình ---
SERVER_HOST = os.getenv("SERVER_HOST", "127.0.0.1")
SERVER_PORT = int(os.getenv("SERVER_PORT", 8000))
# Đã loại bỏ toàn bộ cấu hình Ngrok

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

logging.info(f"Đã thiết lập CUDA_VISIBLE_DEVICES='{os.getenv('CUDA_VISIBLE_DEVICES')}'. Đang cố gắng tải model lên GPU...")

# Import app FastAPI từ file main.py
# Model sẽ được tải ở bước này nếu 'main.py' định nghĩa nó ở global scope
try:
    from main import app
    logging.info("Tải 'main:app' (và model) thành công.")
except Exception as e:
    logging.error(f"❌ LỖI NGHIÊM TRỌNG KHI IMPORT 'main:app': {e}")
    logging.error("Lỗi này có thể xảy ra nếu:")
    logging.error("  1. Bị 'Circular Import' (ví dụ: file 'main.py' của bạn đang import file 'server.py' này).")
    logging.error("  2. Bạn không có GPU/Driver CUDA tương thích.")
    logging.error("  3. PyTorch (với CUDA) chưa được cài đặt đúng cách.")
    logging.error("  4. Có lỗi cú pháp trong file 'main.py'.")
    logging.error("Vui lòng kiểm tra lại file 'main.py' và thiết lập CUDA của bạn. Thoát...")
    exit(1) # Thoát nếu không import được app

def main():
    """Hàm chính để cấu hình và chạy server (chỉ local)"""

    # Toàn bộ logic Ngrok đã được loại bỏ

    print("\n" + "="*80)
    print("🚀 Bắt đầu khởi chạy Palm Tree Counting API Server (SAHI Edition)...")
    print(f"🏠 Server đang chạy nội bộ tại: http://{SERVER_HOST}:{SERVER_PORT}")
    print(f"   -> API docs (SwaggerUI) tại: http://{SERVER_HOST}:{SERVER_PORT}/docs")
    print(f"🔌 Chế độ: Local-only (Đã ép buộc sử dụng CUDA)")
    print("="*80 + "\n")

    try:
        # Chạy server.
        # Nếu bạn muốn tự động tải lại khi code thay đổi (dev mode), thêm: reload=True
        uvicorn.run("main:app", host=SERVER_HOST, port=SERVER_PORT, log_level="info", reload=False)
    except KeyboardInterrupt:
        print("\n👋 Đã nhận tín hiệu dừng (Ctrl+C).")
    finally:
        print("\n👋 Server đã dừng. Tạm biệt!")

if __name__ == "__main__":
    main()

