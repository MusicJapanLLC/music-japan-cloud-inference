import threading
import unittest
from retry_budget import Budget


class RollingBudgetAcceptance(unittest.TestCase):
    def test_capacity_and_boundary(self):
        b = Budget(2, 10)
        self.assertTrue(b.try_acquire(0))
        self.assertTrue(b.try_acquire(1))
        self.assertFalse(b.try_acquire(1.5))
        self.assertTrue(b.try_acquire(10))
        self.assertFalse(b.try_acquire(10.5))
        self.assertTrue(b.try_acquire(11))

    def test_reject_invalid_parameters(self):
        for limit, window in [(0, 10), (2, 0), (-1, 10), (2, -1)]:
            with self.assertRaises(ValueError):
                Budget(limit, window)

    def test_monotonic_and_reset(self):
        b = Budget(1, 10)
        self.assertTrue(b.try_acquire(5))
        with self.assertRaises(ValueError):
            b.try_acquire(4)
        b.reset()
        self.assertTrue(b.try_acquire(-100))

    def test_thread_safety(self):
        b = Budget(4, 100)
        gate = threading.Barrier(24)
        results = []
        def worker():
            gate.wait()
            results.append(b.try_acquire(25))
        threads = [threading.Thread(target=worker) for _ in range(24)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=3)
            self.assertFalse(t.is_alive())
        self.assertEqual(sum(results), 4)


if __name__ == '__main__':
    unittest.main()
