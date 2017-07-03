import unittest


class TestSceneseg(unittest.TestCase):
  # def setUp(self):
  #   print(' setUp func for testCase called')

  # def tearDown(self):
  #   print(' tear down func called')

  def test_upper(self):
    self.assertEqual('foo'.upper(), 'FOO')

  def test_isupper(self):
    self.assertTrue('FOO'.isupper())
    self.assertFalse('fff'.isupper())

  def test_split(self):
    s = "hello world"
    self.assertEqual(s.split(), ['hello', 'world'])
    # check that s.split fails when separtor is not a string
    with self.assertRaises(TypeError):
      s.split(2)

if __name__ == "__main__": unittest.main()