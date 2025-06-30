import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from funsearch import function
from funsearch.datadriven import Dataset, enhanced_dataset_evaluator


@dataclass 
class SkeletonInfo:
    index: int
    skeleton: function.Skeleton
    optimal_params: np.ndarray
    score: float
    mse: float
    description: str


class OneDimensionalPlotComponent:
    def __init__(self, dataset: Dataset):
        if dataset.inputs.shape[1] != 1:
            raise ValueError(f"OneDimensionalPlotComponent only supports 1D input data, got {dataset.inputs.shape[1]}D")
        self.dataset = dataset
        self.skeletons: Dict[int, SkeletonInfo] = {}
        self.selected_skeletons: List[int] = []
        
    def add_skeleton(self, index: int, skeleton: function.Skeleton, description: str = "") -> bool:
        try:
            result = enhanced_dataset_evaluator(skeleton, self.dataset)
            self.skeletons[index] = SkeletonInfo(
                index=index,
                skeleton=skeleton,
                optimal_params=result.optimal_params,
                score=result.score,
                mse=result.mse,
                description=description or f"Function {index}"
            )
            return True
        except Exception as e:
            print(f"Error adding skeleton {index}: {e}")
            return False
    
    def get_available_skeletons(self) -> List[Tuple[int, str, float]]:
        return [(info.index, info.description, info.score) 
                for info in self.skeletons.values()]
    
    def select_skeletons(self, indices: List[int]) -> None:
        self.selected_skeletons = [i for i in indices if i in self.skeletons]
    
    def generate_predictions(self, skeleton_info: SkeletonInfo, 
                           params: Optional[np.ndarray] = None) -> np.ndarray:
        if params is None:
            params = skeleton_info.optimal_params
            
        x_input = self.dataset.inputs.flatten()
        
        try:
            return skeleton_info.skeleton(x_input, params)
        except Exception as e:
            print(f"Error generating predictions: {e}")
            return np.array([])
    
    def create_plot_data(self, param_adjustments: Optional[Dict[int, np.ndarray]] = None) -> Dict[str, Any]:
        if param_adjustments is None:
            param_adjustments = {}
            
        plot_data = {
            'x_data': self.dataset.inputs.flatten(),
            'y_actual': self.dataset.outputs,
            'functions': []
        }
        
        for idx in self.selected_skeletons:
            skeleton_info = self.skeletons[idx]
            params = param_adjustments.get(idx, skeleton_info.optimal_params)
            
            y_pred = self.generate_predictions(skeleton_info, params)
            if len(y_pred) > 0:
                mse = np.mean((y_pred - self.dataset.outputs) ** 2)
                plot_data['functions'].append({
                    'index': idx,
                    'description': skeleton_info.description,
                    'y_pred': y_pred,
                    'params': params,
                    'mse': mse,
                    'original_mse': skeleton_info.mse
                })
        
        return plot_data
    
    def get_param_bounds(self, skeleton_idx: int) -> List[Tuple[float, float]]:
        if skeleton_idx not in self.skeletons:
            return []
        
        params = self.skeletons[skeleton_idx].optimal_params
        return [(param - abs(param) * 2.0 if param != 0 else -10.0,
                 param + abs(param) * 2.0 if param != 0 else 10.0) 
                for param in params]


def create_matplotlib_plot(plot_data: Dict[str, Any]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x_data = plot_data['x_data']
    y_actual = plot_data['y_actual']
    
    ax.scatter(x_data, y_actual, color='black', s=50, alpha=0.7, label='Actual Data', zorder=5)
    
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
    
    for i, func_data in enumerate(plot_data['functions']):
        color = colors[i % len(colors)]
        mse_text = f"MSE: {func_data['mse']:.4f}"
        label = f"{func_data['description']} ({mse_text})"
        
        ax.plot(x_data, func_data['y_pred'], color=color, linewidth=2, 
                label=label, alpha=0.8)
    
    ax.set_xlabel('Input')
    ax.set_ylabel('Output')
    ax.set_title('Function Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig