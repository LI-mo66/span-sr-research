#!/usr/bin/env bash
# Run in the rental platform's root web terminal to authorize this user's EXISTING public key.
# Contains a public key only; no password/private key. Does not alter sshd policy.
set -euo pipefail
umask 077
mkdir -p /root/.ssh
touch /root/.ssh/authorized_keys
key='ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIOhucejfzL5QU9nX49k6J05ONJ8TMSGgckFNOPrWgczz 3083975874@qq.com'
if ! grep -qxF "$key" /root/.ssh/authorized_keys; then
    printf '%s\n' "$key" >> /root/.ssh/authorized_keys
fi
chmod 700 /root/.ssh
chmod 600 /root/.ssh/authorized_keys
printf 'Public key configured. Existing authorized keys preserved.\n'
