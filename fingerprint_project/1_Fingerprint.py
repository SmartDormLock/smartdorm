import time
import serial
import adafruit_fingerprint
import lgpio  # 🔥 ganti ini

DATA_FILE = "finger_users.txt"
RELAY_PIN = 27

# ================= RELAY SETUP =================
h = lgpio.gpiochip_open(0)
lgpio.gpio_claim_output(h, RELAY_PIN)
lgpio.gpio_write(h, RELAY_PIN, 0)  # default LOCK

# ================= FILE MANAGEMENT =================
def load_users():
    users = {}

    try:
        with open(DATA_FILE, "r") as f:
            for line in f:
                id, name = line.strip().split(",")
                users[int(id)] = name
    except FileNotFoundError:
        pass

    return users


def save_users(users):
    with open(DATA_FILE, "w") as f:
        for id, name in users.items():
            f.write(f"{id},{name}\n")


# ================= SERIAL SETUP =================
uart = serial.Serial("/dev/ttyUSB0", baudrate=57600, timeout=2)
time.sleep(2)

finger = adafruit_fingerprint.Adafruit_Fingerprint(uart)

if finger.verify_password() != adafruit_fingerprint.OK:
    print("❌ Sensor fingerprint TIDAK terdeteksi")
    raise SystemExit
else:
    print("✅ Sensor fingerprint TERDETEKSI")


users = load_users()

# ================= RELAY CONTROL =================
def buka_pintu():
    print("Membuka pintu...")
    lgpio.gpio_write(h, RELAY_PIN, 1)


def kunci_pintu():
    print("Mengunci pintu...")
    lgpio.gpio_write(h, RELAY_PIN, 0)


def open_door():
    buka_pintu()
    time.sleep(5)
    kunci_pintu()
    print("Pintu terkunci kembali")


# ================= MENU =================
def show_menu():
    print("\n====== MENU ======")
    print("[1] Enroll sidik jari (multi template)")
    print("[2] Scan / Verifikasi sidik jari")
    print("[3] Hapus ID tertentu")
    print("[4] Lihat user")
    print("[5] Hapus SEMUA data")
    print("==================")


# ================= BACA JARI =================
def read_fingerprint():
    while True:
        i = finger.get_image()

        if i == adafruit_fingerprint.OK:
            return True

        elif i == adafruit_fingerprint.NOFINGER:
            time.sleep(0.1)

        else:
            return False


# ================= TUNGGU JARI DILEPAS =================
def wait_finger_release():
    while finger.get_image() != adafruit_fingerprint.NOFINGER:
        time.sleep(0.1)

    time.sleep(3)


# ================= ENROLL =================
def enroll_fingerprint(location, name=None):

    if location < 1 or location > 127:
        print("❌ ID tidak valid")
        return

    if name is None:
        name = input("Masukkan nama pemilik sidik jari: ")

    max_attempts = 3
    attempt = 0

    while attempt < max_attempts:

        print("\nTempelkan sidik jari 1...")
        if not read_fingerprint():
            print("Sensor error")
            return

        if finger.image_2_tz(1) != adafruit_fingerprint.OK:
            print("Gagal membaca sidik jari")
            attempt += 1
            wait_finger_release()
            continue

        wait_finger_release()

        print("Tempelkan sidik jari 2...")
        if not read_fingerprint():
            print("Sensor error")
            return

        if finger.image_2_tz(2) != adafruit_fingerprint.OK:
            print("Gagal membaca sidik jari kedua")
            attempt += 1
            wait_finger_release()
            continue

        if finger.create_model() == adafruit_fingerprint.OK:

            if finger.store_model(location) == adafruit_fingerprint.OK:

                users[location] = name
                save_users(users)

                print("\n🎉 ENROLL BERHASIL!")
                print(f"ID   : {location}")
                print(f"Nama : {name}")

            else:
                print("Gagal menyimpan ke sensor")

            wait_finger_release()
            return

        else:
            attempt += 1
            print(f"Sidik jari tidak cocok, coba lagi ({attempt}/{max_attempts})")
            wait_finger_release()

    print("❌ Enroll gagal setelah 3 percobaan")


# ================= SCAN =================
def scan_fingerprint():

    print("\n=== MODE SCAN ===")

    if not read_fingerprint():
        return

    if finger.image_2_tz(1) != adafruit_fingerprint.OK:
        print("❌ Gagal proses gambar")
        return

    if finger.finger_fast_search() != adafruit_fingerprint.OK:
        print("\n❌ AKSES DITOLAK")
        print("Sidik jari tidak dikenal")
        wait_finger_release()
        return

    id = finger.finger_id
    confidence = finger.confidence

    print("\n==============================")

    if id in users:
        print("✅ AKSES DITERIMA")
        print("Nama :", users[id])
        open_door()  # 🔥 tambahan relay
    else:
        print("⚠️ ID dikenali tapi nama tidak ditemukan")

    print("ID :", id)
    print("Confidence :", confidence)

    print("==============================")

    wait_finger_release()


# ================= DELETE =================
def delete_fingerprint(location):

    if finger.delete_model(location) == adafruit_fingerprint.OK:

        if location in users:
            del users[location]
            save_users(users)

        print("🗑️ ID berhasil dihapus")

    else:
        print("❌ Gagal hapus ID")


# ================= LIST USERS =================
def list_users():

    print("\n===== USER TERDAFTAR =====")

    if not users:
        print("Belum ada user")
        return

    for id, name in users.items():
        print(f"ID {id} : {name}")


# ================= CLEAR DATABASE =================
def clear_database():

    if finger.empty_library() == adafruit_fingerprint.OK:
        users.clear()
        save_users(users)
        print("🧹 Semua data fingerprint dihapus")

    else:
        print("❌ Gagal hapus database")


# ================= MAIN =================
try:
    while True:

        show_menu()
        cmd = input("Pilih menu: ")

        if cmd == "1":
            name = input("Masukkan nama: ")
            count = int(input("Mau enroll berapa kali? "))

            current_id = max(users.keys(), default=0) + 1

            for i in range(count):
                print(f"\nEnroll ke-{i+1} (ID {current_id})")

                enroll_fingerprint(current_id, name)

                users[current_id] = name
                save_users(users)

                current_id += 1

            print(f"\n✅ {name} berhasil enroll {count} template")

        elif cmd == "2":
            scan_fingerprint()

        elif cmd == "3":
            location = int(input("Masukkan ID yang mau dihapus: "))
            delete_fingerprint(location)

        elif cmd == "4":
            list_users()

        elif cmd == "5":
            clear_database()

        elif cmd.lower() == "q":
            print("Keluar...")
            break

        else:
            print("Menu tidak valid")

except KeyboardInterrupt:
    print("\nProgram dihentikan")

finally:
    lgpio.gpio_write(h, RELAY_PIN, 0)
    lgpio.gpiochip_close(h)
