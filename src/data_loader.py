import os
import sys
from pathlib import Path
import pandas as pd

# Insert the project’s src directory into sys.path so that simulation_manager can be imported
is_windows = os.name == "nt"
project_root_env = os.getenv("SIM_ROOT")
if project_root_env:
    PROJECT_ROOT = Path(project_root_env).expanduser()
elif is_windows:
    PROJECT_ROOT = Path.cwd()
else:
    PROJECT_ROOT = Path.home() / "Research_Project"

SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from simulation_manager import SimulationManager


class DataLoader:
    """
    Carga los datasets de simulación para tablas 2 y 3 usando SimulationManager.
    """

    def __init__(self):
        # Definir DATA_DIR y crear el SimulationManager
        self.DATA_DIR = PROJECT_ROOT / "simulation_data"
        self.sim_manager = SimulationManager(base_dir=self.DATA_DIR)

        # En sistemas UNIX, corregir rutas si es necesario
        if not is_windows:
            self.sim_manager.fix_paths(self.DATA_DIR)

    def load_table2(self):
        """
        Carga todos los datasets correspondientes a la 'table2'.
        Devuelve:
            - datasets: lista de objetos dgp (data generating process)
            - metadata: lista de diccionarios con metadatos de cada dataset
            - configs_df: DataFrame con los config_id disponibles
        """
        available_configs = self.sim_manager.list_available_datasets(table="table2")
        datasets = []
        metadata = []

        print("Cargando Table 2:")
        config_ids = available_configs["config_id"].tolist()
        total = len(config_ids)
        print(f"  {total} configuraciones encontradas.")
        for i, config_id in enumerate(config_ids, start=1):
            try:
                dgp, meta = self.sim_manager.load_dataset(config_id=config_id)
                datasets.append(dgp)
                metadata.append(meta)
                method = meta.get("method", "N/A")
                items = meta.get("item_count", "N/A")
                common = meta.get("common_items", "N/A")
                examinees = meta.get("examinee_count", "N/A")
                print(f"    [{i:2d}/{total}] {config_id}: method={method}, items={items}, CI={common}, N={examinees}")
            except Exception as e:
                print(f"    [{i:2d}/{total}] {config_id} - ERROR: {e}")

        print(f"  → Table 2: cargados {len(datasets)}/{total} datasets.\n")
        return datasets, metadata, available_configs

    def load_table3(self):
        """
        Carga los datasets correspondientes a la 'table3' según la lista fija de configuraciones multi-población.
        Devuelve:
            - datasets: lista de objetos dgp
            - metadata: lista de diccionarios con metadatos de cada dataset
            - configs_df: DataFrame con los config_id disponibles
        """
        # Definir las configuraciones de Table 3 (mu_list, sigma2, common_items)
        multi_pop_configs = [
            {"mu_list": "[-0.3, 0.3]", "sigma2": 0.7, "common_items": 5},
            {"mu_list": "[-0.5, 0.5]", "sigma2": 0.5, "common_items": 5},
            {"mu_list": "[-0.7, 0.7]", "sigma2": 0.3, "common_items": 5},
            {"mu_list": "[-0.9, 0.9]", "sigma2": 0.1, "common_items": 5},
            {"mu_list": "[-0.3, 0.3]", "sigma2": 0.7, "common_items": 0},
            {"mu_list": "[-0.5, 0.5]", "sigma2": 0.5, "common_items": 0},
            {"mu_list": "[-0.7, 0.7]", "sigma2": 0.3, "common_items": 0},
            {"mu_list": "[-0.9, 0.9]", "sigma2": 0.1, "common_items": 0},
        ]

        available_configs = self.sim_manager.list_available_datasets(table="table3")
        datasets = []
        metadata = []

        print("Cargando Table 3:")
        total = len(multi_pop_configs)
        print(f"  {total} configuraciones multi-población a cargar.")
        for i, cfg in enumerate(multi_pop_configs, start=1):
            subset = available_configs[
                (available_configs["mu_list"] == cfg["mu_list"])
                & (available_configs["sigma2"] == cfg["sigma2"])
                & (available_configs["common_items"] == cfg["common_items"])
            ]
            if not subset.empty:
                config_id = subset.iloc[0]["config_id"]
                try:
                    dgp, meta = self.sim_manager.load_dataset(config_id=config_id)
                    datasets.append(dgp)
                    metadata.append(meta)
                    print(
                        f"    [{i:2d}/{total}] {config_id}: μ={cfg['mu_list']}, σ²={cfg['sigma2']}, CI={cfg['common_items']}"
                    )
                except Exception as e:
                    print(f"    [{i:2d}/{total}] {config_id} - ERROR: {e}")
            else:
                print(
                    f"    [{i:2d}/{total}] NO ENCONTRADO: μ={cfg['mu_list']}, σ²={cfg['sigma2']}, CI={cfg['common_items']}"
                )

        print(f"  → Table 3: cargados {len(datasets)}/{total} datasets.\n")
        return datasets, metadata, available_configs

    def get_item_ids_per_block(self, dgp):
        """
        Devuelve lista de arrays con item_ids por bloque/test si existen.
        Debe ser estable a través de tests para que los ítems comunes tengan el MISMO id.
        Si no existen, devuelve None y se avisará para usar linking mean/sd como fallback.
        """
        # Caso 1: atributo estándar
        if hasattr(dgp, "item_ids") and isinstance(dgp.item_ids, (list, tuple)):
            return [pd.Index(ids).to_numpy() for ids in dgp.item_ids]

        # Caso 2: datasets con 'meta_by_block' o similar (ajusta si tu SimulationManager lo guarda así)
        if hasattr(dgp, "meta_by_block"):
            ids = []
            for blk in dgp.meta_by_block:
                if "item_ids" in blk:
                    ids.append(pd.Index(blk["item_ids"]).to_numpy())
                else:
                    return None
            return ids

        # No hay forma segura de identificar ítems comunes
        return None