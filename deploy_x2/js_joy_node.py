"""Publish /joy straight from /dev/input/js0 — no ros-humble-joy needed.

The bundled DualSense pairs over Bluetooth to SoC0, so joy_node would have to
live there, but SoC0 has no ros-humble-joy and installing needs sudo. The Linux
joystick API is a fixed 8-byte binary struct, so reading it directly is a few
lines and costs nothing.

It doubles as the button-mapping verifier: --probe prints every event with its
index as you press, so the ❌⭕️🔺🟥 -> buttons[0..3] mapping the controller
hardcodes gets checked against this pad instead of assumed. Getting ⭕️ (damping,
the safe exit) confused with ❌ (zero torque) would drop a standing robot.

  python3 js_joy_node.py --probe          # press keys, see indices
  python3 js_joy_node.py                  # publish /joy at 50 Hz
"""

from __future__ import annotations

import argparse
import os
import struct

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy

# struct js_event { __u32 time; __s16 value; __u8 type; __u8 number; }
JS_EVENT = "IhBB"
JS_SIZE = struct.calcsize(JS_EVENT)
JS_BUTTON, JS_AXIS, JS_INIT = 0x01, 0x02, 0x80

# What the official controller hardcodes (motion_control_node.cc joyCallback).
EXPECTED = {0: "cross ❌ PASSIVE 零力矩", 1: "circle ⭕️ DAMPING 阻尼-安全出口",
            2: "triangle 🔺 JOINT 位控", 3: "square 🟥 RL 策略"}


class JsJoy(Node):
    def __init__(self, dev, probe):
        super().__init__("js_joy_node")
        self.fd = os.open(dev, os.O_RDONLY | os.O_NONBLOCK)
        self.probe = probe
        self.buttons = [0] * 16
        self.axes = [0.0] * 8
        self.pub = None if probe else self.create_publisher(Joy, "/joy", 10)
        self.create_timer(0.02, self._tick)          # 50 Hz
        if probe:
            self.get_logger().info(
                f"PROBE {dev}: press each button; nothing is published.\n"
                "  compare against what the controller assumes:\n  " +
                "\n  ".join(f"buttons[{i}] = {n}" for i, n in EXPECTED.items()))
        else:
            self.get_logger().info(f"publishing /joy from {dev} at 50 Hz")

    def _drain(self):
        while True:
            try:
                buf = os.read(self.fd, JS_SIZE)
            except BlockingIOError:
                return
            if not buf or len(buf) < JS_SIZE:
                return
            _t, val, typ, num = struct.unpack(JS_EVENT, buf)
            init = bool(typ & JS_INIT)          # synthetic startup event
            typ &= ~JS_INIT
            if typ == JS_BUTTON and num < len(self.buttons):
                self.buttons[num] = 1 if val else 0
                if self.probe and not init:
                    tag = EXPECTED.get(num, "(控制器未用到)")
                    self.get_logger().info(
                        f"BUTTON [{num}] {'按下' if val else '松开':4s} -> "
                        f"控制器认为这是 {tag}")
            elif typ == JS_AXIS and num < len(self.axes):
                self.axes[num] = val / 32767.0

    def _tick(self):
        self._drain()
        if self.pub is None:
            return
        m = Joy()
        m.header.stamp = self.get_clock().now().to_msg()
        m.axes = [float(a) for a in self.axes]
        m.buttons = [int(b) for b in self.buttons]
        self.pub.publish(m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", default="/dev/input/js0")
    ap.add_argument("--probe", action="store_true",
                    help="print button indices instead of publishing")
    args = ap.parse_args()
    rclpy.init()
    n = JsJoy(args.dev, args.probe)
    try:
        rclpy.spin(n)
    except KeyboardInterrupt:
        pass
    finally:
        os.close(n.fd)
        n.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
