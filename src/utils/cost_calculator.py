from typing import Dict, Optional
from dataclasses import dataclass
import math
from datetime import datetime
import json
import os

@dataclass
class MachineConfig:
    """Configuration for Google Cloud machine types"""
    name: str
    cpu_count: int
    memory_gb: float
    gpu_type: Optional[str]
    gpu_count: int
    cost_per_hour: float
    
class VertexAICostCalculator:
    """Calculator for estimating Vertex AI training costs"""
    
    # Machine configurations and pricing (US Central1)
    MACHINE_CONFIGS = {
        'n1-standard-4': MachineConfig(
            name='n1-standard-4',
            cpu_count=4,
            memory_gb=15,
            gpu_type=None,
            gpu_count=0,
            cost_per_hour=0.19
        ),
        'n1-standard-8-t4': MachineConfig(
            name='n1-standard-8-T4',
            cpu_count=8,
            memory_gb=30,
            gpu_type='NVIDIA_TESLA_T4',
            gpu_count=1,
            cost_per_hour=0.19 + 0.35  # Base + GPU
        ),
        'n1-standard-8-v100': MachineConfig(
            name='n1-standard-8-V100',
            cpu_count=8,
            memory_gb=30,
            gpu_type='NVIDIA_TESLA_V100',
            gpu_count=1,
            cost_per_hour=0.19 + 2.48  # Base + GPU
        )
    }

    def __init__(self, model_params: Dict):
        """Initialize with model parameters"""
        self.model_params = model_params
        self.hidden_size = model_params.get('hidden_size', 256)
        self.num_layers = model_params.get('num_liquid_layers', 3)
        self.batch_size = model_params.get('batch_size', 32)
        
    def estimate_training_time(self, dataset_size: int, epochs: int) -> Dict:
        """Estimate training time for different machine configurations"""
        estimates = {}
        
        for machine_name, config in self.MACHINE_CONFIGS.items():
            # Basic computation time estimation
            base_time = self._calculate_base_time(dataset_size, epochs)
            
            # Apply machine-specific speedup factors
            if config.gpu_type == 'NVIDIA_TESLA_V100':
                speedup = 8.0  # V100 is roughly 8x faster than CPU
            elif config.gpu_type == 'NVIDIA_TESLA_T4':
                speedup = 4.0  # T4 is roughly 4x faster than CPU
            else:
                speedup = 1.0  # CPU baseline
                
            estimated_hours = base_time / speedup
            estimated_cost = estimated_hours * config.cost_per_hour
            
            estimates[machine_name] = {
                'estimated_hours': round(estimated_hours, 2),
                'cost_per_hour': round(config.cost_per_hour, 2),
                'total_cost': round(estimated_cost, 2),
                'machine_config': config
            }
            
        return estimates
    
    def _calculate_base_time(self, dataset_size: int, epochs: int) -> float:
        """Calculate base training time in hours"""
        # Simplified computation time estimation
        samples_per_second = 10  # Base processing rate
        total_samples = dataset_size * epochs
        
        # Factor in model complexity
        complexity_factor = (self.hidden_size / 256) * (self.num_layers / 3)
        
        # Calculate total hours
        total_seconds = (total_samples / samples_per_second) * complexity_factor
        return total_seconds / 3600  # Convert to hours
    
    def print_cost_summary(self, dataset_size: int, epochs: int):
        """Print a formatted cost summary"""
        estimates = self.estimate_training_time(dataset_size, epochs)
        
        print("\n=== Vertex AI Training Cost Estimates ===")
        print(f"Dataset size: {dataset_size:,} samples")
        print(f"Epochs: {epochs}")
        print(f"Model: {self.hidden_size} hidden size, {self.num_layers} layers")
        print("\nEstimates by machine type:")
        
        for machine_name, estimate in estimates.items():
            print(f"\n{machine_name}:")
            print(f"  Training time: {estimate['estimated_hours']:.1f} hours")
            print(f"  Cost per hour: ${estimate['cost_per_hour']:.2f}")
            print(f"  Total cost: ${estimate['total_cost']:.2f}")
            if estimate['machine_config'].gpu_type:
                print(f"  GPU: {estimate['machine_config'].gpu_type}")
    
    def run(self, dataset_size: int, epochs: int, save_report: bool = True) -> Dict:
        """
        Run cost analysis and optionally save report
        
        Args:
            dataset_size: Number of training samples
            epochs: Number of training epochs
            save_report: Whether to save report to file
            
        Returns:
            Dictionary containing cost estimates
        """
        estimates = self.estimate_training_time(dataset_size, epochs)
        self.print_cost_summary(dataset_size, epochs)
        
        if save_report:
            self.save_report(dataset_size, epochs, estimates)
            
        return estimates
    
    def save_report(self, dataset_size: int, epochs: int, estimates: Dict):
        """Save detailed cost report to file"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_dir = 'cost_reports'
        os.makedirs(report_dir, exist_ok=True)
        
        report = {
            'timestamp': timestamp,
            'model_config': {
                'hidden_size': self.hidden_size,
                'num_layers': self.num_layers,
                'batch_size': self.batch_size
            },
            'training_config': {
                'dataset_size': dataset_size,
                'epochs': epochs,
                'total_samples': dataset_size * epochs
            },
            'estimates': {
                machine: {
                    k: str(v) if k == 'machine_config' else v 
                    for k, v in data.items()
                }
                for machine, data in estimates.items()
            }
        }
        
        # Save JSON report
        report_path = os.path.join(report_dir, f'cost_report_{timestamp}.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
            
        # Save markdown report
        md_report = self._generate_markdown_report(report)
        md_path = os.path.join(report_dir, f'cost_report_{timestamp}.md')
        with open(md_path, 'w') as f:
            f.write(md_report)
            
        print(f"\nReports saved to:")
        print(f"- JSON: {report_path}")
        print(f"- Markdown: {md_path}")
    
    def _generate_markdown_report(self, report: Dict) -> str:
        """Generate formatted markdown report"""
        md = [
            "# Vertex AI Training Cost Report",
            f"Generated on: {report['timestamp']}",
            
            "\n## Model Configuration",
            f"- Hidden Size: {report['model_config']['hidden_size']}",
            f"- Number of Layers: {report['model_config']['num_layers']}",
            f"- Batch Size: {report['model_config']['batch_size']}",
            
            "\n## Training Configuration",
            f"- Dataset Size: {report['training_config']['dataset_size']:,} samples",
            f"- Epochs: {report['training_config']['epochs']}",
            f"- Total Samples: {report['training_config']['total_samples']:,}",
            
            "\n## Cost Estimates"
        ]
        
        # Add estimates for each machine type
        for machine, data in report['estimates'].items():
            md.extend([
                f"\n### {machine}",
                f"- Training Time: {data['estimated_hours']:.1f} hours",
                f"- Cost per Hour: ${data['cost_per_hour']:.2f}",
                f"- Total Cost: ${data['total_cost']:.2f}"
            ])
            
            # Add GPU info if present
            machine_config = eval(data['machine_config'])
            if machine_config.gpu_type:
                md.append(f"- GPU: {machine_config.gpu_type}")
        
        return '\n'.join(md)


def main():
    """Main function to run cost calculations with default parameters"""
    # Fix the import path
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    from src.config.config import Config
    
    # Initialize calculator with default config
    config = Config()
    calculator = VertexAICostCalculator(config.to_dict())
    
    # Default training scenarios
    scenarios = [
        {
            'name': 'Small Dataset',
            'dataset_size': 10000,
            'epochs': 50
        },
        {
            'name': 'Medium Dataset',
            'dataset_size': 100000,
            'epochs': 100
        },
        {
            'name': 'Large Dataset',
            'dataset_size': 1000000,
            'epochs': 100
        }
    ]
    
    print("\n=== ScholarAI Training Cost Analysis ===")
    print("Running cost estimates for different training scenarios...")
    
    for scenario in scenarios:
        print(f"\n>> Scenario: {scenario['name']}")
        calculator.run(
            dataset_size=scenario['dataset_size'],
            epochs=scenario['epochs'],
            save_report=True
        )
        
    print("\nAnalysis complete! Check the 'cost_reports' directory for detailed reports.")

if __name__ == "__main__":
    main()