#!/usr/bin/env python3
"""Phi accrual failure detector — adaptive failure detection for distributed systems.

One file. Zero deps. Does one thing well.

Used in Akka, Cassandra, and other distributed systems. Instead of binary
alive/dead, outputs a suspicion level (phi) that increases with silence.
Based on Hayashibara et al. (2004).
"""
import math, time, sys

class PhiAccrualDetector:
    def __init__(self, threshold=8.0, max_samples=1000, min_std=500):
        self.threshold = threshold
        self.max_samples = max_samples
        self.min_std = min_std  # minimum stddev in ms
        self.intervals = []
        self.last_heartbeat = None

    def heartbeat(self, now=None):
        now = now or time.time() * 1000  # ms
        if self.last_heartbeat is not None:
            interval = now - self.last_heartbeat
            self.intervals.append(interval)
            if len(self.intervals) > self.max_samples:
                self.intervals.pop(0)
        self.last_heartbeat = now

    def _mean(self):
        return sum(self.intervals) / len(self.intervals) if self.intervals else 0

    def _stddev(self):
        if len(self.intervals) < 2:
            return self.min_std
        mean = self._mean()
        var = sum((x - mean) ** 2 for x in self.intervals) / (len(self.intervals) - 1)
        return max(math.sqrt(var), self.min_std)

    def phi(self, now=None):
        """Calculate phi (suspicion level). Higher = more suspicious."""
        if self.last_heartbeat is None or len(self.intervals) < 1:
            return 0.0
        now = now or time.time() * 1000
        elapsed = now - self.last_heartbeat
        mean = self._mean()
        std = self._stddev()
        # Phi = -log10(P(X > elapsed)) assuming normal distribution
        # Using complementary CDF approximation
        y = (elapsed - mean) / std
        # Approximation of log10(1 - Phi(y)) where Phi is normal CDF
        if y <= -5: return 0.0
        if y >= 10: return 30.0
        # Use error function approximation
        e = 1 / (1 + 0.3275911 * abs(y / math.sqrt(2)))
        t = e
        erfx = 1 - (0.254829592*t - 0.284496736*t**2 + 1.421413741*t**3 
                     - 1.453152027*t**4 + 1.061405429*t**5) * math.exp(-y*y/2)
        if y < 0: erfx = 1 - erfx
        p_later = 1.0 - erfx / 2 if y >= 0 else erfx / 2
        p_later = max(p_later, 1e-15)
        return -math.log10(p_later)

    def is_available(self, now=None):
        return self.phi(now) < self.threshold

def main():
    print("Phi Accrual Failure Detector\n")
    fd = PhiAccrualDetector(threshold=8.0)
    
    # Simulate regular heartbeats at ~1000ms intervals
    t = 0
    for i in range(20):
        t += 1000 + (i % 3) * 50  # slight jitter
        fd.heartbeat(t)
    
    print(f"After 20 regular heartbeats (mean={fd._mean():.0f}ms, std={fd._stddev():.0f}ms):")
    for delay in [1000, 2000, 3000, 5000, 8000, 15000]:
        p = fd.phi(t + delay)
        status = "✓ available" if p < fd.threshold else "✗ suspected"
        print(f"  +{delay:5d}ms: phi={p:6.2f} {status}")
    
    # Simulate node going down then recovering
    print("\nSimulation: node slows then recovers")
    fd2 = PhiAccrualDetector(threshold=8.0)
    t = 0
    for i in range(10):
        t += 1000
        fd2.heartbeat(t)
    # Slow period
    for i in range(3):
        t += 3000
        p = fd2.phi(t)
        fd2.heartbeat(t)
        print(f"  t={t:6d}ms: phi={p:.2f} (slow heartbeat)")
    # Recovery
    for i in range(5):
        t += 1000
        p = fd2.phi(t)
        fd2.heartbeat(t)
        print(f"  t={t:6d}ms: phi={p:.2f} (recovered)")

if __name__ == "__main__":
    main()
