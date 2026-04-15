"""
IP Blocker via Windows Firewall
================================
Blocks/unblocks IPs using Windows Firewall rules (netsh advfirewall).
Requires Administrator privileges.
"""
import subprocess
import time
import threading
from typing import Set, Dict


# IPs that should NEVER be blocked
WHITELIST = {
    "127.0.0.1",
    "0.0.0.0",
    "255.255.255.255",
    "::1",
}

# Private network prefixes to never block
PRIVATE_PREFIXES = ("192.168.", "10.", "172.16.", "172.17.", "172.18.",
                    "172.19.", "172.20.", "172.21.", "172.22.", "172.23.",
                    "172.24.", "172.25.", "172.26.", "172.27.", "172.28.",
                    "172.29.", "172.30.", "172.31.")

# Firewall rule name prefix
RULE_PREFIX = "MambaIDS_Block_"


def is_safe_ip(ip: str) -> bool:
    """Check if an IP is safe (should not be blocked)."""
    if ip in WHITELIST:
        return True
    if ip.startswith(PRIVATE_PREFIXES):
        return True
    return False


class IPBlocker:
    """Manages IP blocking via Windows Firewall."""

    def __init__(self, auto_unblock_seconds: int = 300):
        """
        Args:
            auto_unblock_seconds: Automatically unblock IPs after this many seconds.
                                  Set to 0 to disable auto-unblock.
        """
        self.auto_unblock_seconds = auto_unblock_seconds
        self.blocked_ips: Dict[str, float] = {}  # ip -> block timestamp
        self.block_log: list = []  # List of {ip, action, timestamp, reason}
        self._lock = threading.Lock()

        # Start auto-unblock thread if enabled
        if auto_unblock_seconds > 0:
            self._unblock_thread = threading.Thread(
                target=self._auto_unblock_loop, daemon=True
            )
            self._unblock_thread.start()

    def block_ip(self, ip: str, reason: str = "Mamba IDS detection") -> bool:
        """
        Block an IP address using Windows Firewall.

        Returns True if blocked, False if skipped (whitelisted or already blocked).
        """
        if is_safe_ip(ip):
            self._log(ip, "SKIPPED", f"Whitelisted: {reason}")
            return False

        with self._lock:
            if ip in self.blocked_ips:
                return False  # Already blocked

            rule_name = f"{RULE_PREFIX}{ip.replace('.', '_')}"

            try:
                # Add inbound block rule
                cmd_in = [
                    "netsh", "advfirewall", "firewall", "add", "rule",
                    f"name={rule_name}_IN",
                    "dir=in", "action=block",
                    f"remoteip={ip}",
                    "enable=yes"
                ]
                subprocess.run(cmd_in, capture_output=True, timeout=10,
                             creationflags=subprocess.CREATE_NO_WINDOW)

                # Add outbound block rule
                cmd_out = [
                    "netsh", "advfirewall", "firewall", "add", "rule",
                    f"name={rule_name}_OUT",
                    "dir=out", "action=block",
                    f"remoteip={ip}",
                    "enable=yes"
                ]
                subprocess.run(cmd_out, capture_output=True, timeout=10,
                             creationflags=subprocess.CREATE_NO_WINDOW)

                self.blocked_ips[ip] = time.time()
                self._log(ip, "BLOCKED", reason)
                return True

            except Exception as e:
                self._log(ip, "ERROR", f"Failed to block: {e}")
                return False

    def unblock_ip(self, ip: str) -> bool:
        """Remove firewall block rule for an IP."""
        with self._lock:
            if ip not in self.blocked_ips:
                return False

            rule_name = f"{RULE_PREFIX}{ip.replace('.', '_')}"

            try:
                for suffix in ["_IN", "_OUT"]:
                    cmd = [
                        "netsh", "advfirewall", "firewall", "delete", "rule",
                        f"name={rule_name}{suffix}"
                    ]
                    subprocess.run(cmd, capture_output=True, timeout=10,
                                 creationflags=subprocess.CREATE_NO_WINDOW)

                del self.blocked_ips[ip]
                self._log(ip, "UNBLOCKED", "Auto-unblock timeout")
                return True

            except Exception as e:
                self._log(ip, "ERROR", f"Failed to unblock: {e}")
                return False

    def unblock_all(self):
        """Remove all MambaIDS firewall rules."""
        with self._lock:
            for ip in list(self.blocked_ips.keys()):
                self.unblock_ip(ip)

            # Also cleanup any orphaned rules
            try:
                cmd = [
                    "netsh", "advfirewall", "firewall", "show", "rule",
                    f"name=all", "dir=in"
                ]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=10,
                                      creationflags=subprocess.CREATE_NO_WINDOW)
                # Delete any remaining MambaIDS rules
                for line in result.stdout.split('\n'):
                    if RULE_PREFIX in line and "Rule Name:" in line:
                        rule = line.split(":")[-1].strip()
                        subprocess.run(
                            ["netsh", "advfirewall", "firewall", "delete", "rule", f"name={rule}"],
                            capture_output=True, timeout=10,
                            creationflags=subprocess.CREATE_NO_WINDOW
                        )
            except Exception:
                pass

    def get_blocked_ips(self) -> list:
        """Get list of currently blocked IPs with timestamps."""
        with self._lock:
            return [
                {"ip": ip, "blocked_at": time.strftime("%H:%M:%S", time.localtime(ts)),
                 "seconds_ago": int(time.time() - ts)}
                for ip, ts in self.blocked_ips.items()
            ]

    def get_log(self, max_entries: int = 20) -> list:
        """Get recent block/unblock log entries."""
        return self.block_log[-max_entries:]

    def _log(self, ip: str, action: str, reason: str):
        """Add entry to block log."""
        self.block_log.append({
            "timestamp": time.strftime("%H:%M:%S"),
            "ip": ip,
            "action": action,
            "reason": reason
        })
        # Keep log bounded
        if len(self.block_log) > 100:
            self.block_log = self.block_log[-50:]

    def _auto_unblock_loop(self):
        """Background thread to auto-unblock IPs after timeout."""
        while True:
            time.sleep(30)  # Check every 30 seconds
            now = time.time()
            with self._lock:
                expired = [
                    ip for ip, ts in self.blocked_ips.items()
                    if now - ts > self.auto_unblock_seconds
                ]
            for ip in expired:
                self.unblock_ip(ip)
