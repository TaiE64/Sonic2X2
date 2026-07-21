"""Query / switch the X2 system state (AimDK developer mode).

  Ready | Business            native software in charge
  Develop_MC                  native motion control OFF -> we drive the joints
  (also: Develop_Audio_Linux, Develop_Audio_ROS, Develop_Nav)

Per the AimDK docs: always switch back to Ready (or reboot) when done.

Needs the FULL aimdk_msgs (the v0.9.0.4 underlay ships only 18 srv types and
lacks these services):
  source <AIMDK_INSTALL>/setup.bash   # the aimdk_msgs underlay on the robot

Usage:
  python3 set_mode.py                 # query only
  python3 set_mode.py --state Develop_MC
  python3 set_mode.py --state Ready
"""

import argparse

import rclpy
from rclpy.node import Node

from aimdk_msgs.srv import GetSystemState, MigrateSystemState

GET = "/aimdk_5Fmsgs/srv/GetSystemState"
SET = "/aimdk_5Fmsgs/srv/MigrateSystemState"


def query(n):
    cli = n.create_client(GetSystemState, GET)
    if not cli.wait_for_service(timeout_sec=5.0):
        raise RuntimeError(f"{GET} unavailable")
    fut = cli.call_async(GetSystemState.Request())
    rclpy.spin_until_future_complete(n, fut, timeout_sec=8.0)
    r = fut.result()
    if r is None:
        raise RuntimeError("no response from GetSystemState")
    return r.cur_state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", default=None)
    args = ap.parse_args()

    rclpy.init()
    n = Node("x2_set_mode")
    try:
        before = query(n)
        print(f"current: {before!r}")
        if not args.state:
            return
        cli = n.create_client(MigrateSystemState, SET)
        if not cli.wait_for_service(timeout_sec=5.0):
            raise RuntimeError(f"{SET} unavailable")
        req = MigrateSystemState.Request()
        req.state = args.state
        fut = cli.call_async(req)
        rclpy.spin_until_future_complete(n, fut, timeout_sec=15.0)
        resp = fut.result()
        print(f"migrate -> {args.state!r}: {resp.header if resp else 'NO RESPONSE'}")
        after = query(n)
        print(f"now: {after!r}  {'OK' if after == args.state else 'MISMATCH'}")
    finally:
        n.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
