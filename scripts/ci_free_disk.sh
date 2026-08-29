#!/usr/bin/env bash
# Free space on GitHub-hosted Ubuntu runners before Debug/coverage/ASan
# builds that link hundreds of test binaries.
set -u
echo "=== Disk before cleanup ==="
df -h
sudo rm -rf /usr/share/dotnet /usr/local/lib/android /opt/ghc \
    /usr/local/share/powershell /usr/share/swift \
    /opt/hostedtoolcache/CodeQL /opt/hostedtoolcache/go || true
if command -v docker >/dev/null 2>&1; then
    sudo docker image prune -af >/dev/null 2>&1 || true
fi
echo "=== Disk after cleanup ==="
df -h
