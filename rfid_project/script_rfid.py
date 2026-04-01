# -*- coding: utf-8 -*-

import time
import logging
from rc522_spi_library import RC522SPILibrary, StatusCodes

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

DATA_FILE = "rfid_cards.txt"


# ================= FILE MANAGEMENT =================
def load_cards():
    cards = {}

    try:
        with open(DATA_FILE, "r") as f:
            for line in f:
                parts = line.strip().split(",")

                if len(parts) == 2:
                    uid, name = parts
                    cards[uid] = name

    except FileNotFoundError:
        pass

    return cards


def save_cards(cards):
    with open(DATA_FILE, "w") as f:
        for uid, name in cards.items():
            f.write(f"{uid},{name}\n")


# ================= MENU =================
def show_menu():
    print("\n====== MENU RFID ======")
    print("[1] Daftarkan kartu (Enroll)")
    print("[2] Scan / Verifikasi kartu")
    print("[3] Hapus UID")
    print("[4] Lihat semua UID")
    print("[5] Keluar")
    print("=======================")


# ================= READ CARD =================
def read_card(reader):

    print("\nTempelkan kartu RFID...")

    while True:

        status, _ = reader.request()

        if status == StatusCodes.OK:

            status, uid = reader.anticoll()

            if status == StatusCodes.OK:

                uid_str = ":".join([f"{i:02X}" for i in uid])

                print(f"\nUID kartu: {uid_str}")

                # tunggu kartu dilepas dulu
                while True:

                    status, _ = reader.request()

                    if status != StatusCodes.OK:
                        break

                    time.sleep(0.1)

                # cooldown 3 detik
                print("Menunggu 3 detik sebelum scan berikutnya...\n")
                time.sleep(3)

                return uid_str

        time.sleep(0.1)

# ================= ENROLL =================
def enroll_card(reader, cards):

    name = input("Masukkan nama pemilik kartu: ")

    uid = read_card(reader)

    if uid is None:
        return

    # kalau nama belum ada
    if name not in cards:

        cards[name] = [uid]
        print("Kartu pertama berhasil didaftarkan")

    else:

        if uid in cards[name]:
            print("Kartu ini sudah terdaftar untuk user ini")

        elif len(cards[name]) >= 2:
            print("⚠️ Setiap orang hanya boleh 2 kartu")

        else:
            cards[name].append(uid)
            print("Kartu kedua berhasil didaftarkan")

    save_cards(cards)

# ================= VERIFY =================
def verify_card(reader, cards):

    print("\n=== MODE SCAN KARTU ===")

    uid = read_card(reader)

    if uid is None:
        return

    print("\n==============================")

    if uid in cards:

        print("✅ AKSES DITERIMA")
        print(f"Nama : {cards[uid]}")
        print(f"UID  : {uid}")

    else:

        print("❌ AKSES DITOLAK")
        print("Kartu tidak dikenal")

    print("==============================")

# ================= DELETE =================
def delete_card(cards):

    uid = input("\nMasukkan UID yang akan dihapus: ")

    if uid in cards:
        del cards[uid]
        save_cards(cards)
        print("✅ UID berhasil dihapus")

    else:
        print("❌ UID tidak ditemukan")


# ================= LIST =================
def list_cards(cards):

    print("\n===== DAFTAR KARTU TERDAFTAR =====")

    if not cards:
        print("Belum ada kartu terdaftar")
        return

    for i, (uid, name) in enumerate(cards.items(), 1):
        print(f"{i}. {name}  |  {uid}")


# ================= MAIN =================
def main():

    print("Starting RFID System...")

    reader = None

    try:

        reader = RC522SPILibrary(rst_pin=25)

        cards = load_cards()

        while True:

            show_menu()

            choice = input("Pilih menu: ")

            if choice == "1":
                enroll_card(reader, cards)

            elif choice == "2":
                verify_card(reader, cards)

            elif choice == "3":
                delete_card(cards)

            elif choice == "4":
                list_cards(cards)

            elif choice == "5":
                print("Keluar program...")
                break

            else:
                print("Menu tidak valid!")

    except Exception as e:
        logging.error(f"Error: {e}")

    except KeyboardInterrupt:
        print("\nProgram dihentikan")

    finally:
        if reader:
            reader.cleanup()
            print("RC522 resources released successfully.")


if __name__ == "__main__":
    main()