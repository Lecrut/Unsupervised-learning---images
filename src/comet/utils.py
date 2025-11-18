from __future__ import annotations
from datetime import datetime
import json
import os
import logging
from typing import Optional, Dict, Any, Union
from pathlib import Path
from comet_ml import Experiment
import numpy as np
import matplotlib.pyplot as plt
from dotenv import load_dotenv


def check_comet_env():
    """Sprawdza czy plik .env istnieje i informuje użytkownika"""
    if not os.path.exists('.env'):
        print("Brak pliku .env")
        print("Aby używać Comet ML:")
        print("   1. Skopiuj .env.template jako .env")
        print("   2. Uzupełnij swoje dane w pliku .env")
        print("   3. Lub zostaw USE_COMET = False w następnej komórce")
    else:
        print("Plik .env znaleziony")


class CometModel:
    """Klasa do logowania eksperymentów z Comet ML i lokalnie"""
    
    def __init__(self, experiment_name, use_comet=True, use_local=True):
        self.experiment_name = experiment_name
        self.use_comet = use_comet
        self.use_local = use_local
        
        self.comet_experiment = None
        self.local_logger = None
        
        if use_comet:
            try:
                
                load_dotenv()
                
                api_key = os.getenv("COMET_API_KEY")
                project_name = os.getenv("COMET_PROJECT_NAME")
                workspace = os.getenv("COMET_WORKSPACE")
                
                if not all([api_key, project_name, workspace]):
                    print("Brak zmiennych środowiskowych dla Comet ML")
                    self.use_comet = False
                else:
                    self.comet_experiment = Experiment(
                        api_key=api_key,
                        project_name=project_name,
                        workspace=workspace
                    )
                    print(f"Comet ML aktywny: {project_name}")
            except Exception as e:
                print(f"Comet ML niedostępny: {e}")
                self.use_comet = False
        
        if use_local:
            self.local_logger = LocalLogger(experiment_name)
            print(f"Lokalny logger aktywny: {experiment_name}")
    
    def log_parameters(self, params: Dict):
        if self.comet_experiment:
            self.comet_experiment.log_parameters(params)
        if self.local_logger:
            self.local_logger.log_parameters(params)
    
    def log_metrics(self, metrics: Dict, step: Optional[int] = None):
        if self.comet_experiment:
            self.comet_experiment.log_metrics(metrics, step=step)
        if self.local_logger:
            self.local_logger.log_metrics(metrics, step=step)
    
    def log_metric(self, name: str, value: float, step: Optional[int] = None):
        if self.comet_experiment:
            self.comet_experiment.log_metric(name, value, step=step)
        if self.local_logger:
            self.local_logger.log_metrics({name: value}, step=step)
    
    def log_image(self, image, name: str, step: Optional[int] = None):
        if self.comet_experiment:
            self.comet_experiment.log_image(image, name=name, step=step)
    
    def log_figure(self, figure_name: str, figure):
        if self.comet_experiment:
            self.comet_experiment.log_figure(figure_name, figure)
    
    def log_model(self, model_name: str, model_path: str):
        if self.comet_experiment:
            self.comet_experiment.log_model(model_name, model_path)
    
    def end(self):
        if self.comet_experiment:
            self.comet_experiment.end()
        if self.local_logger:
            self.local_logger.end()


class ExperimentLogger(CometModel):
    """Alias dla zachowania kompatybilności"""
    pass

class LocalLogger:
    """
    Prosty logger lokalny jako zamiennik dla Comet ML
    """
    
    def __init__(self, experiment_name: str = "autoencoder_experiment", 
                 log_dir: str = "local_logs"):
        self.experiment_name = experiment_name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_dir = self.log_dir / f"{experiment_name}_{self.timestamp}"
        self.experiment_dir.mkdir(exist_ok=True)
        
        self.metrics = {}
        self.parameters = {}
        self.step_data = []
        
        print(f"Lokalne logowanie w: {self.experiment_dir}")
    
    def log_parameters(self, params: Dict[str, Any]):
        """Loguje parametry eksperymentu"""
        self.parameters.update(params)
        
        # Zapisz do pliku
        params_file = self.experiment_dir / "parameters.json"
        with open(params_file, 'w') as f:
            json.dump(self.parameters, f, indent=2, default=str)
        
        print("Parametry zapisane lokalnie:")
        for key, value in params.items():
            print(f"  {key}: {value}")
    
    def log_metric(self, name: str, value: float, step: Optional[int] = None):
        """Loguje pojedynczą metrykę"""
        if name not in self.metrics:
            self.metrics[name] = []
        
        metric_entry = {
            'value': value,
            'step': step if step is not None else len(self.metrics[name]),
            'timestamp': datetime.now().isoformat()
        }
        
        self.metrics[name].append(metric_entry)
        self.step_data.append({
            'metric': name,
            'value': value,
            'step': metric_entry['step'],
            'timestamp': metric_entry['timestamp']
        })
    
    def log_metrics(self, metrics_dict: Dict[str, float], step: Optional[int] = None):
        """Loguje wiele metryk jednocześnie"""
        for name, value in metrics_dict.items():
            self.log_metric(name, value, step)
    
    def log_figure(self, name: str, figure=None):
        """Zapisuje wykres jako obraz"""
        if figure is None:
            figure = plt.gcf()
        
        figure_path = self.experiment_dir / f"{name}.png"
        figure.savefig(figure_path, dpi=150, bbox_inches='tight')
        print(f"Wykres zapisany: {figure_path}")
    
    def log_image(self, image_array, name: str):
        """Zapisuje obraz z numpy array"""
        import numpy as np
        
        if image_array.max() <= 1.0:
            image_array = (image_array * 255).astype(np.uint8)
        
        image_path = self.experiment_dir / f"{name}.png"
        
        if len(image_array.shape) == 3:  # RGB
            plt.figure(figsize=(8, 8))
            plt.imshow(image_array)
            plt.axis('off')
            plt.title(name)
            plt.savefig(image_path, dpi=150, bbox_inches='tight')
            plt.close()
        
        print(f"Obraz zapisany: {image_path}")
    
    def save_summary(self):
        """Zapisuje podsumowanie eksperymentu"""
        summary = {
            'experiment_name': self.experiment_name,
            'timestamp': self.timestamp,
            'parameters': self.parameters,
            'metrics_summary': {}
        }
        
        for metric_name, metric_data in self.metrics.items():
            values = [entry['value'] for entry in metric_data]
            summary['metrics_summary'][metric_name] = {
                'count': len(values),
                'min': min(values),
                'max': max(values),
                'final': values[-1] if values else None,
                'mean': sum(values) / len(values) if values else None
            }
        
        summary_file = self.experiment_dir / "summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        metrics_file = self.experiment_dir / "metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2, default=str)
        
        print(f"Podsumowanie zapisane: {summary_file}")
        return summary
    
    def plot_metrics(self):
        """Tworzy wykresy wszystkich metryk"""
        if not self.metrics:
            print("Brak metryk do wykreślenia")
            return
        
        n_metrics = len(self.metrics)
        if n_metrics == 1:
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            axes = [ax]
        else:
            cols = min(2, n_metrics)
            rows = (n_metrics + cols - 1) // cols
            fig, axes = plt.subplots(rows, cols, figsize=(12, 6*rows))
            if n_metrics > 1:
                axes = axes.flatten()
            else:
                axes = [axes]
        
        for i, (metric_name, metric_data) in enumerate(self.metrics.items()):
            if i < len(axes):
                steps = [entry['step'] for entry in metric_data]
                values = [entry['value'] for entry in metric_data]
                
                axes[i].plot(steps, values, 'b-', linewidth=2)
                axes[i].set_title(f'{metric_name}')
                axes[i].set_xlabel('Step')
                axes[i].set_ylabel('Value')
                axes[i].grid(True, alpha=0.3)
        
        # Usuń puste subploty
        for i in range(n_metrics, len(axes)):
            fig.delaxes(axes[i])
        
        plt.tight_layout()
        
        metrics_plot_path = self.experiment_dir / "metrics_plot.png"
        plt.savefig(metrics_plot_path, dpi=150, bbox_inches='tight')
        print(f"Wykres metryk zapisany: {metrics_plot_path}")
        
        plt.show()
    
    def end(self):
        """Kończy eksperyment i zapisuje wszystkie dane"""
        summary = self.save_summary()
        self.plot_metrics()
        
        print(f"\nEksperyment zakończony!")
        print(f"Wszystkie pliki w: {self.experiment_dir}")
        print(f"Zalogowane metryki: {list(self.metrics.keys())}")
        
        return summary