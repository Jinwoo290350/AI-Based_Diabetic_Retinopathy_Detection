#!/bin/bash

# ตรวจสอบว่าไฟล์ไบนารีของ Python 3.13 มีอยู่หรือไม่
if command -v python3.13 >/dev/null 2>&1; then
    PYTHON_BIN=$(which python3.13)
    echo "พบ Python 3.13 ที่: $PYTHON_BIN"
    
    echo "กำลังลบไฟล์ไบนารี Python 3.13..."
    sudo rm -f "$PYTHON_BIN"
else
    echo "ไม่พบไฟล์ไบนารี Python 3.13 ใน PATH"
fi

# ลบไดเรกทอรีไลบรารีที่ติดตั้งไว้ (ถ้ามี)
PYTHON_LIB="/usr/local/lib/python3.13"
if [ -d "$PYTHON_LIB" ]; then
    echo "กำลังลบไดเรกทอรีไลบรารี: $PYTHON_LIB"
    sudo rm -rf "$PYTHON_LIB"
else
    echo "ไม่พบไดเรกทอรีไลบรารีที่: $PYTHON_LIB"
fi

echo "ขั้นตอนการลบ Python 3.13.2 เสร็จสิ้นแล้ว"
