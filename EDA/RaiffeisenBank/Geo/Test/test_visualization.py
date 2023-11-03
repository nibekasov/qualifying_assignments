import unittest
from visualization import create_scatter_plot, create_heatmap

class TestVisualization(unittest.TestCase):

    def test_create_scatter_plot(self):
        # Assuming the function create_scatter_plot exists and works as expected
        fig = create_scatter_plot(sample_latitudes, sample_longitudes)
        self.assertIsInstance(fig, expected_figure_type)  # Check if the output is a figure object

    def test_create_heatmap(self):
        # Assuming the function create_heatmap exists and works as expected
        fig = create_heatmap(sample_location_data)
        self.assertIsInstance(fig, expected_figure_type)  # Check if the output is a figure object

# Run the tests
if __name__ == '__main__':
    unittest.main()
