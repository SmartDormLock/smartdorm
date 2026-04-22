import RPi.GPIO as GPIO
import time

RELAY_PIN = 27

GPIO.setmode(GPIO.BCM)
GPIO.setup(RELAY_PIN, GPIO.OUT, initial=GPIO.LOW)

def buka_pintu():
    GPIO.output(RELAY_PIN, GPIO.HIGH)
    time.sleep(0.2)

def kunci_pintu():
    GPIO.output(RELAY_PIN, GPIO.LOW)
    time.sleep(0.2)

print("Test relay + solenoid (FINAL FINAL)")

try:
    while True:
        print("?? BUKA")
        buka_pintu()
        time.sleep(5)

        print("?? KUNCI")
        kunci_pintu()
        time.sleep(5)

except KeyboardInterrupt:
    print("\nStop")

finally:
    print("Pastikan terkunci...")
    kunci_pintu()
    time.sleep(1)
    GPIO.cleanup()
