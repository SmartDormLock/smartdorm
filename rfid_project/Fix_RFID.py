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
                id, name, uid = line.strip().split(",")
                id = int(id)

                if id not in cards:
                    cards[id] = {
                        "name": name,
                        "uids": [uid]
                    }
                else:
                    cards[id]["uids"].append(uid)

    except FileNotFoundError:
        pass

    return cards


def save_cards(cards):
    with open(DATA_FILE, "w") as f:
        for id, data in cards.items():
            for uid in data["uids"]:
                f.write(f"{id},{data['name']},{uid}\n")


# ================= MENU =================
def show_menu():
    print("\n====== MENU RFID ======")
    print("[1] Enroll kartu (max 2)")
    print("[2] Scan / Verifikasi kartu")
    print("[3] Hapus ID tertentu")
    print("[4] Lihat user")
    print("[5] Hapus SEMUA data")
    print("[q] Keluar")
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
                print(f"UID kartu: {uid_str}")

                # tunggu kartu dilepas
                while True:
                    status, _ = reader.request()
                    if status != StatusCodes.OK:
                        break
                    time.sleep(0.1)

                time.sleep(1)
                return uid_str

        time.sleep(0.1)


# ================= ENROLL =================
def enroll_card(reader, cards):

    name = input("Masukkan nama: ")

    new_id = max(cards.keys(), default=0) + 1
    print(f"ID user: {new_id}")

    cards[new_id] = {
        "name": name,
        "uids": []
    }

    while True:

        # LIMIT 2 KARTU
        if len(cards[new_id]["uids"]) >= 2:
            print("⚠️ Maksimal hanya 2 kartu per user")
            break

        uid = read_card(reader)

        # CEK DUPLIKAT GLOBAL
        for id, data in cards.items():
            if uid in data["uids"]:
                print("❌ UID sudah terdaftar di user lain!")
                return

        # CEK DUPLIKAT DI USER INI
        if uid in cards[new_id]["uids"]:
            print("⚠️ UID sudah ditambahkan")
            continue

        cards[new_id]["uids"].append(uid)
        print(f"✅ Kartu ke-{len(cards[new_id]['uids'])} berhasil ditambahkan")

        if len(cards[new_id]["uids"]) < 2:
            lanjut = input("Tambah kartu kedua? (y/n): ")
            if lanjut.lower() != "y":
                break

    save_cards(cards)

    print("\n🎉 ENROLL SELESAI")
    print(f"Nama : {name}")
    print(f"ID   : {new_id}")
    print(f"Total UID : {len(cards[new_id]['uids'])}")


# ================= VERIFY =================
def verify_card(reader, cards):

    print("\n=== MODE SCAN RFID ===")

    uid = read_card(reader)

    for id, data in cards.items():
        if uid in data["uids"]:
            print("\n==============================")
            print("✅ AKSES DITERIMA")
            print(f"Nama : {data['name']}")
            print(f"ID   : {id}")
            print("==============================")
            return

    print("\n❌ AKSES DITOLAK")
    print("Kartu tidak dikenal")


# ================= DELETE BY ID =================
def delete_card(cards):

    try:
        id = int(input("Masukkan ID yang mau dihapus: "))
    except ValueError:
        print("❌ ID harus angka")
        return

    if id in cards:
        del cards[id]
        save_cards(cards)
        print("🗑️ User berhasil dihapus")
    else:
        print("❌ ID tidak ditemukan")


# ================= LIST USERS =================
def list_cards(cards):

    print("\n===== USER TERDAFTAR =====")

    if not cards:
        print("Belum ada user")
        return

    for id, data in cards.items():
        print(f"\nID {id} : {data['name']}")
        print(f"UID : {', '.join(data['uids'])}")


# ================= CLEAR DATABASE =================
def clear_database(reader, cards):

    confirm = input("Yakin hapus semua data? (y/n): ")

    if confirm.lower() == "y":
        cards.clear()
        save_cards(cards)
        print("🧹 Semua data RFID dihapus")
    else:
        print("Dibatalkan")


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
                clear_database(reader, cards)

            elif choice.lower() == "q":
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
            print("RC522 cleanup selesai")


if __name__ == "__main__":
    main()
