"""
Network Packet Sniffer & Flow Feature Extractor
================================================
Captures live network packets using Scapy, groups them into flows,
and extracts CICFlowMeter-style features for the Mamba IDS model.

Requires: scapy, admin/root privileges
"""
import time
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from scapy.all import sniff, IP, TCP, UDP, ICMP, conf
    SCAPY_AVAILABLE = True
except ImportError:
    SCAPY_AVAILABLE = False


@dataclass
class FlowRecord:
    """Tracks statistics for a single network flow."""
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    protocol: int
    start_time: float = 0.0
    last_time: float = 0.0

    # Packet counts
    fwd_packets: int = 0
    bwd_packets: int = 0

    # Byte counts
    fwd_bytes: int = 0
    bwd_bytes: int = 0

    # Packet lengths
    fwd_lengths: list = field(default_factory=list)
    bwd_lengths: list = field(default_factory=list)

    # Inter-arrival times
    fwd_iats: list = field(default_factory=list)
    bwd_iats: list = field(default_factory=list)
    flow_iats: list = field(default_factory=list)

    # Timestamps for IAT calculation
    last_fwd_time: float = 0.0
    last_bwd_time: float = 0.0
    last_packet_time: float = 0.0

    # TCP flags
    syn_count: int = 0
    fin_count: int = 0
    rst_count: int = 0
    psh_count: int = 0
    ack_count: int = 0
    urg_count: int = 0

    # Header lengths
    fwd_header_len: int = 0
    bwd_header_len: int = 0


def _safe_mean(lst):
    return np.mean(lst) if lst else 0.0

def _safe_std(lst):
    return np.std(lst) if lst else 0.0

def _safe_max(lst):
    return max(lst) if lst else 0.0

def _safe_min(lst):
    return min(lst) if lst else 0.0


def flow_to_features(flow: FlowRecord) -> np.ndarray:
    """
    Convert a FlowRecord to a 78-dimensional feature vector
    matching the CICIDS2017 feature format.

    Features we can compute are filled in; the rest are zero-padded.
    """
    duration = max(flow.last_time - flow.start_time, 1e-6)
    total_packets = flow.fwd_packets + flow.bwd_packets
    total_bytes = flow.fwd_bytes + flow.bwd_bytes
    all_lengths = flow.fwd_lengths + flow.bwd_lengths

    features = np.zeros(77, dtype=np.float32)

    # 0: Destination Port
    features[0] = flow.dst_port

    # 1: Flow Duration (microseconds)
    features[1] = duration * 1e6

    # 2-3: Total Fwd/Bwd Packets
    features[2] = flow.fwd_packets
    features[3] = flow.bwd_packets

    # 4-5: Total Length of Fwd/Bwd Packets
    features[4] = flow.fwd_bytes
    features[5] = flow.bwd_bytes

    # 6-9: Fwd Packet Length (Max, Min, Mean, Std)
    features[6] = _safe_max(flow.fwd_lengths)
    features[7] = _safe_min(flow.fwd_lengths)
    features[8] = _safe_mean(flow.fwd_lengths)
    features[9] = _safe_std(flow.fwd_lengths)

    # 10-13: Bwd Packet Length (Max, Min, Mean, Std)
    features[10] = _safe_max(flow.bwd_lengths)
    features[11] = _safe_min(flow.bwd_lengths)
    features[12] = _safe_mean(flow.bwd_lengths)
    features[13] = _safe_std(flow.bwd_lengths)

    # 14-15: Flow Bytes/s, Flow Packets/s
    features[14] = total_bytes / duration
    features[15] = total_packets / duration

    # 16-19: Flow IAT (Mean, Std, Max, Min)
    features[16] = _safe_mean(flow.flow_iats)
    features[17] = _safe_std(flow.flow_iats)
    features[18] = _safe_max(flow.flow_iats)
    features[19] = _safe_min(flow.flow_iats)

    # 20-23: Fwd IAT (Total, Mean, Std, Max, Min)
    features[20] = sum(flow.fwd_iats) if flow.fwd_iats else 0
    features[21] = _safe_mean(flow.fwd_iats)
    features[22] = _safe_std(flow.fwd_iats)
    features[23] = _safe_max(flow.fwd_iats)

    # 24-27: Bwd IAT (Total, Mean, Std, Max, Min)
    features[24] = sum(flow.bwd_iats) if flow.bwd_iats else 0
    features[25] = _safe_mean(flow.bwd_iats)
    features[26] = _safe_std(flow.bwd_iats)
    features[27] = _safe_max(flow.bwd_iats)

    # 28-29: Fwd PSH Flags, Bwd PSH Flags
    features[28] = flow.psh_count
    features[29] = 0

    # 30-31: Fwd URG Flags, Bwd URG Flags
    features[30] = flow.urg_count
    features[31] = 0

    # 32-33: Fwd Header Length, Bwd Header Length
    features[32] = flow.fwd_header_len
    features[33] = flow.bwd_header_len

    # 34-35: Fwd Packets/s, Bwd Packets/s
    features[34] = flow.fwd_packets / duration
    features[35] = flow.bwd_packets / duration

    # 36-40: Packet Length (Min, Max, Mean, Std, Variance)
    features[36] = _safe_min(all_lengths)
    features[37] = _safe_max(all_lengths)
    features[38] = _safe_mean(all_lengths)
    features[39] = _safe_std(all_lengths)
    features[40] = _safe_std(all_lengths) ** 2

    # 41-48: TCP Flag counts
    features[41] = flow.fin_count
    features[42] = flow.syn_count
    features[43] = flow.rst_count
    features[44] = flow.psh_count
    features[45] = flow.ack_count
    features[46] = flow.urg_count
    features[47] = 0  # CWE
    features[48] = 0  # ECE

    # 49: Down/Up Ratio
    features[49] = flow.bwd_packets / max(flow.fwd_packets, 1)

    # 50-52: Average Packet Size, Fwd Avg Segment Size, Bwd Avg Segment Size
    features[50] = _safe_mean(all_lengths)
    features[51] = _safe_mean(flow.fwd_lengths)
    features[52] = _safe_mean(flow.bwd_lengths)

    # 53-58: Bulk features (zero-padded — hard to compute without full flow)
    # 59-62: Subflow features
    features[59] = flow.fwd_packets
    features[60] = flow.fwd_bytes
    features[61] = flow.bwd_packets
    features[62] = flow.bwd_bytes

    # 63-64: Init Win Bytes (Forward/Backward) — would need TCP window
    features[63] = 0
    features[64] = 0

    # 65-66: Act data pkt fwd, min seg size forward
    features[65] = flow.fwd_packets
    features[66] = _safe_min(flow.fwd_lengths) if flow.fwd_lengths else 0

    # 67-74: Active/Idle Mean, Std, Max, Min
    features[67] = duration * 1e6  # Active mean
    features[68] = 0
    features[69] = duration * 1e6
    features[70] = duration * 1e6
    # Idle stats
    features[71] = 0
    features[72] = 0
    features[73] = 0
    features[74] = 0

    # 75-77: remaining features (zero-padded)
    return features


class NetworkSniffer:
    """
    Live network packet sniffer that groups packets into flows
    and extracts features for the IDS model.
    """

    def __init__(self, interface=None, flow_timeout=30.0, max_flow_duration=120.0):
        """
        Args:
            interface: Network interface to sniff on (None = default)
            flow_timeout: Seconds of inactivity before a flow expires
            max_flow_duration: Maximum flow duration before forcing expiry
        """
        if not SCAPY_AVAILABLE:
            raise ImportError("Scapy is required. Install: pip install scapy")

        self.interface = interface
        self.flow_timeout = flow_timeout
        self.max_flow_duration = max_flow_duration

        self.active_flows: Dict[tuple, FlowRecord] = {}
        self.completed_flows: List[Tuple[FlowRecord, np.ndarray]] = []
        self.flow_lock = threading.Lock()

        self._running = False
        self._thread = None

        # Stats
        self.total_packets = 0
        self.total_flows_completed = 0

    def _flow_key(self, pkt) -> Optional[tuple]:
        """Extract 5-tuple flow key from packet."""
        if not pkt.haslayer(IP):
            return None

        src_ip = pkt[IP].src
        dst_ip = pkt[IP].dst
        proto = pkt[IP].proto

        src_port = 0
        dst_port = 0
        if pkt.haslayer(TCP):
            src_port = pkt[TCP].sport
            dst_port = pkt[TCP].dport
        elif pkt.haslayer(UDP):
            src_port = pkt[UDP].sport
            dst_port = pkt[UDP].dport

        return (src_ip, dst_ip, src_port, dst_port, proto)

    def _process_packet(self, pkt):
        """Process a single captured packet."""
        key = self._flow_key(pkt)
        if key is None:
            return

        self.total_packets += 1
        now = time.time()
        pkt_len = len(pkt)

        src_ip, dst_ip, src_port, dst_port, proto = key
        # Determine direction: forward = same as flow initiator
        reverse_key = (dst_ip, src_ip, dst_port, src_port, proto)

        with self.flow_lock:
            # Check if this is forward or reverse direction
            if key in self.active_flows:
                flow = self.active_flows[key]
                is_forward = True
            elif reverse_key in self.active_flows:
                flow = self.active_flows[reverse_key]
                key = reverse_key
                is_forward = False
            else:
                # New flow
                flow = FlowRecord(
                    src_ip=src_ip, dst_ip=dst_ip,
                    src_port=src_port, dst_port=dst_port,
                    protocol=proto,
                    start_time=now, last_time=now,
                    last_fwd_time=now, last_packet_time=now
                )
                self.active_flows[key] = flow
                is_forward = True

            # Update flow stats
            flow.last_time = now

            # Flow IAT
            if flow.last_packet_time > 0 and flow.last_packet_time != now:
                iat = (now - flow.last_packet_time) * 1e6  # microseconds
                flow.flow_iats.append(iat)
            flow.last_packet_time = now

            if is_forward:
                flow.fwd_packets += 1
                flow.fwd_bytes += pkt_len
                flow.fwd_lengths.append(pkt_len)
                if flow.last_fwd_time > 0 and flow.last_fwd_time != now:
                    flow.fwd_iats.append((now - flow.last_fwd_time) * 1e6)
                flow.last_fwd_time = now
                if pkt.haslayer(IP):
                    flow.fwd_header_len += pkt[IP].ihl * 4
            else:
                flow.bwd_packets += 1
                flow.bwd_bytes += pkt_len
                flow.bwd_lengths.append(pkt_len)
                if flow.last_bwd_time > 0 and flow.last_bwd_time != now:
                    flow.bwd_iats.append((now - flow.last_bwd_time) * 1e6)
                flow.last_bwd_time = now
                if pkt.haslayer(IP):
                    flow.bwd_header_len += pkt[IP].ihl * 4

            # TCP flags
            if pkt.haslayer(TCP):
                flags = pkt[TCP].flags
                if flags & 0x02: flow.syn_count += 1
                if flags & 0x01: flow.fin_count += 1
                if flags & 0x04: flow.rst_count += 1
                if flags & 0x08: flow.psh_count += 1
                if flags & 0x10: flow.ack_count += 1
                if flags & 0x20: flow.urg_count += 1

    def _expire_flows(self):
        """Check for expired flows and move to completed list."""
        now = time.time()
        expired_keys = []

        with self.flow_lock:
            for key, flow in self.active_flows.items():
                idle = now - flow.last_time
                duration = now - flow.start_time

                if idle > self.flow_timeout or duration > self.max_flow_duration:
                    # Only complete flows with at least 2 packets
                    if flow.fwd_packets + flow.bwd_packets >= 2:
                        features = flow_to_features(flow)
                        self.completed_flows.append((flow, features))
                        self.total_flows_completed += 1
                    expired_keys.append(key)

            for key in expired_keys:
                del self.active_flows[key]

    def get_completed_flows(self, max_count=50) -> List[Tuple[FlowRecord, np.ndarray]]:
        """Get and remove completed flows."""
        with self.flow_lock:
            result = self.completed_flows[:max_count]
            self.completed_flows = self.completed_flows[max_count:]
            return result

    def get_stats(self) -> dict:
        """Get sniffer statistics."""
        with self.flow_lock:
            return {
                "total_packets": self.total_packets,
                "active_flows": len(self.active_flows),
                "completed_flows": self.total_flows_completed,
                "pending_flows": len(self.completed_flows),
                "running": self._running,
            }

    def _sniff_loop(self):
        """Background sniffing loop."""
        def packet_callback(pkt):
            if self._running:
                self._process_packet(pkt)

        while self._running:
            try:
                sniff(
                    iface=self.interface,
                    prn=packet_callback,
                    store=False,
                    timeout=5,
                    filter="ip",
                )
                self._expire_flows()
            except Exception as e:
                print(f"[Sniffer] Error: {e}")
                time.sleep(1)

    def start(self):
        """Start packet capture in background thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._sniff_loop, daemon=True)
        self._thread.start()
        print("[Sniffer] Started capturing packets...")

    def stop(self):
        """Stop packet capture."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=10)
        print("[Sniffer] Stopped.")


def get_available_interfaces():
    """Get list of available network interfaces."""
    if not SCAPY_AVAILABLE:
        return ["default"]
    try:
        from scapy.arch.windows import get_windows_if_list
        ifaces = get_windows_if_list()
        return [iface['name'] for iface in ifaces if iface.get('name')]
    except Exception:
        try:
            return [str(i) for i in conf.ifaces.data.values()]
        except Exception:
            return ["default"]
