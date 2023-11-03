import unittest
from data_preprocessing import convert_to_datetime, round_coordinates

class TestDataPreprocessing(unittest.TestCase):

    def test_convert_to_datetime(self):
        # Assuming the function convert_to_datetime exists and works as expected
        self.assertEqual(convert_to_datetime("20230101T0000Z"), expected_datetime_object)

    def test_round_coordinates(self):
        # Assuming the function round_coordinates exists and works as expected
        rounded_lat, rounded_lon = round_coordinates(52.367634, 4.904138)
        self.assertEqual(rounded_lat, expected_rounded_lat)
        self.assertEqual(rounded_lon, expected_rounded_lon)

# Run the tests
if __name__ == '__main__':
    unittest.main()
