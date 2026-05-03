#!/usr/bin/env bash
set -euo pipefail

IFACE="${PIDS_LIDAR_IFACE:-enP8p1s0}"
ADDR="${PIDS_LIDAR_LOCAL_ADDR:-169.254.62.10/16}"

if ! ip link show "$IFACE" >/dev/null 2>&1; then
  echo "LiDAR interface not found: $IFACE" >&2
  exit 1
fi

sudo ip link set "$IFACE" up
if ! ip addr show dev "$IFACE" | grep -q "${ADDR%/*}"; then
  sudo ip addr add "$ADDR" dev "$IFACE"
fi

ip -br addr show "$IFACE"
