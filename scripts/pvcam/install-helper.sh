#!/bin/bash
set -euo pipefail
if [[ $EUID -ne 0 ]]; then
    echo 'Run this installer with sudo.' >&2
    exit 1
fi
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
recovery_user=${SUDO_USER:-}
if [[ ! $recovery_user =~ ^[a-z_][a-z0-9_-]*$ ]] || [[ $recovery_user == root ]]; then
    echo 'Run sudo from the workstation user account.' >&2
    exit 1
fi
id "$recovery_user" >/dev/null
for utility in /usr/bin/python3 /usr/sbin/modprobe /usr/bin/setpci /usr/bin/udevadm /usr/sbin/visudo; do
    test -x "$utility"
done
sudoers_tmp=$(mktemp)
trap 'rm -f "$sudoers_tmp"' EXIT
printf '%s ALL=(root) NOPASSWD: /usr/local/sbin/voxel-recover-pvcam "", /usr/local/sbin/voxel-recover-pvcam --retrain\n' "$recovery_user" > "$sudoers_tmp"
/usr/sbin/visudo -cf "$sudoers_tmp"
install -o root -g root -m 0755 "$script_dir/voxel-recover-pvcam" /usr/local/sbin/voxel-recover-pvcam
install -o root -g root -m 0440 "$sudoers_tmp" /etc/sudoers.d/voxel-pvcam
/usr/sbin/visudo -cf /etc/sudoers.d/voxel-pvcam
echo "Installed PVCAM recovery helper for $recovery_user. No hardware recovery was run."
