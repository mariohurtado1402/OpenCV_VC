import serial
import time

arduino = serial.Serial('/dev/ttyACM0', 9600, timeout=1)
time.sleep(2)

arduino.write(b'1')
print("LED ENCENDIDO")
time.sleep(2)

arduino.write(b'0')
print("LED APAGADO")

arduino.close()
