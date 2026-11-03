#!/bin/sh
set -eu
if [ "$(id -u)" -ne 0 ]; then
    printf '%s\n' 'Run: sudo ./install-udev-rules.sh'
    exit 1
fi
app_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
install -m 0644 "$app_dir/70-sranibro-xr5.rules" /etc/udev/rules.d/70-sranibro-xr5.rules
udevadm control --reload-rules
printf '%s\n' 'XR5 USB access rule installed. Reconnect the headset USB, then run the application as your normal desktop user.'
