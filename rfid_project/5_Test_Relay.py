import RPi.GPIO as GPIO
import time

GPIO.setmode(GPIO.BCM)
GPIO.setup(27, GPIO.OUT)

print("LOW dulu")
GPIO.output(27, GPIO.LOW)
time.sleep(3)

print("HIGH sekarang")
GPIO.output(27, GPIO.HIGH)
time.sleep(3)

GPIO.cleanup()
