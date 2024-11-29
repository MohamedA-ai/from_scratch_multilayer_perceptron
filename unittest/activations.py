import unittest
from src import activation


class TestActivationFunctions(unittest.TestCase):
    def test_sigmoid(self):
        sig = activation.Sigmoid()
        self.assertAlmostEqual(sig(0), 0.5)
        self.assertAlmostEqual(sig.derivate(0), 0.25)

    def test_tanh(self):
        tanh = activation.Tanh()
        self.assertAlmostEqual(tanh(0), 0)
        self.assertAlmostEqual(tanh.derivate(0), 1)

    def test_relu(self):
        relu = activation.ReLU()
        self.assertEqual(relu(0), 0)
        self.assertEqual(relu.derivate(0), 0)


if __name__ == '__main__':
    unittest.main()
