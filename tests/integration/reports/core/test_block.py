from explorica.reports.core.block import Block
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from explorica.types import VisualizationResult


def test_init_with_dict_containing_visualizations():
    """Block normalizes visualizations passed in a dict"""
    fig1, ax = plt.subplots()
    try:
        fig2 = go.Figure(data=go.Bar(y=[1, 2, 3]))
        
        cfg_dict = {
            "title": "Visual Test",
            "visualizations": [fig1, fig2],
            "metrics": []
        }
        block = Block(cfg_dict)
    
        # Check, that visualizations are normalized
        for vis in block.block_config.visualizations:
            # normalize_visualization must return an object of type VisualizationResult
            assert isinstance(vis, VisualizationResult)
    finally:
        plt.close(fig1)