import usb.core
import can
import threading

dev = usb.core.find(idVendor=0x04d8, idProduct=0x0053)
if dev is None:
    raise RuntimeError("Waveshare CANalyst-II not found (vendor=0x04d8, product=0x0053) — check USB connection")
if dev.is_kernel_driver_active(0): # type: ignore
    dev.detach_kernel_driver(0) # type: ignore
if dev.is_kernel_driver_active(1): # type: ignore
    dev.detach_kernel_driver(1) # type: ignore

hw_bus = can.Bus(interface='canalystii', channel=1, bitrate=1000000)
virt_bus = can.Bus(interface='socketcan', channel='vcan0')

def hw_to_virt():
    while True:
        msg = hw_bus.recv()
        if msg:
            msg.channel = None
            virt_bus.send(msg)

def virt_to_hw():
    while True:
        msg = virt_bus.recv()
        if msg:
            msg.channel = None
            hw_bus.send(msg)

threading.Thread(target=hw_to_virt, daemon=True).start()
threading.Thread(target=virt_to_hw, daemon=True).start()

print("Bridge running. vcan0 <-> Waveshare channel 1")
try:
    while True:
        threading.Event().wait(1)
except KeyboardInterrupt:
    pass