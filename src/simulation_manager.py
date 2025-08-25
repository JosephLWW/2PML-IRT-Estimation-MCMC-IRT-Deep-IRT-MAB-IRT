"""
Simulation Manager for dataset generation and management
"""
import itertools
import os
import numpy as np
import pandas as pd
import pickle
from datetime import datetime
from pathlib import Path
from dgp import DGP2PLM

class SimulationManager:
    def __init__(self, base_dir="simulation_data"):
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)
        self.metadata_path = os.path.join(base_dir, "metadata.csv")
        
        # Define paths for combined files
        self.table2_path = os.path.join(base_dir, "table2_combined.pkl")
        self.table3_path = os.path.join(base_dir, "table3_combined.pkl")
        
        # Load existing metadata or create new
        if os.path.exists(self.metadata_path):
            self.metadata = pd.read_csv(self.metadata_path)
            
            # Ensure 'table' column exists (for backward compatibility)
            if 'table' not in self.metadata.columns:
                print("Adding 'table' column to existing metadata")
                # Determine table based on file path
                self.metadata['table'] = 'unknown'
                if 'file_path' in self.metadata.columns:
                    self.metadata.loc[self.metadata['file_path'].str.contains('table2'), 'table'] = 'table2'
                    self.metadata.loc[self.metadata['file_path'].str.contains('table3'), 'table'] = 'table3'
                # Save updated metadata
                self.metadata.to_csv(self.metadata_path, index=False)
        else:
            self.metadata = pd.DataFrame(columns=[
                'config_id', 'table', 'method', 'item_count', 'common_items', 
                'examinee_count', 'seed', 'multi_pop', 'creation_date', 
                'file_path', 'total_items', 'total_examinees', 'num_tests',
                'mu_list', 'sigma2'
            ])
    
    def fix_paths(self, base_dir=None):
        """
        ✅ Enhanced path normalization that properly handles cross-platform paths
        """
        if 'file_path' not in self.metadata.columns:
            print("⚠️ No 'file_path' column found in metadata.")
            return

        base = Path(base_dir) if base_dir else Path(self.base_dir)
        
        print(f"🔧 Corrigiendo rutas en metadata...")
        print(f"ANTES:")
        print(self.metadata['file_path'].head(3))
        
        def normalize_path_enhanced(path_str):
            """Enhanced path normalization with better error handling"""
            if not isinstance(path_str, str):
                return path_str
                
            # Convert all separators to forward slashes first
            normalized = path_str.replace('\\', '/')
            
            # Remove duplicate simulation_data directories
            if 'simulation_data/simulation_data' in normalized:
                normalized = normalized.replace('simulation_data/simulation_data', 'simulation_data')
            
            # Extract filename if it's a complex path
            if '/' in normalized:
                filename = normalized.split('/')[-1]
            else:
                filename = normalized
                
            # Ensure we have a .pkl file
            if not filename.endswith('.pkl'):
                return str(path_str)  # Return original if not a pickle file
                
            # Construct proper path using base directory
            proper_path = base / filename
            return str(proper_path)

        # Apply the enhanced normalization
        self.metadata['file_path'] = self.metadata['file_path'].apply(normalize_path_enhanced)
        
        print(f"DESPUÉS:")
        print(self.metadata['file_path'].head(3))
        
        # Save the corrected metadata
        self.metadata.to_csv(self.metadata_path, index=False)
        print(f"✅ Fixed {len(self.metadata)} file paths and saved metadata")

    def generate_config_id(self, config):
        """Generate a unique config ID for a configuration"""
        method_code = 'sys' if config['method'] == 'systematic' else 'rnd'
        
        if config.get('multi_pop', False):
            mu_str = '_'.join([str(m) for m in config['mu_list']])
            return f"{method_code}_i{config['item_count']}_c{config['common_items']}_e{config['examinee_count']}_s{config['seed']}_mp_mu{mu_str}_sig{config['sigma2']}"
        else:
            return f"{method_code}_i{config['item_count']}_c{config['common_items']}_e{config['examinee_count']}_s{config['seed']}"
    
    def generate_all_datasets(self, methods, item_counts, common_items, examinee_counts, seeds, multi_pop_params=None):
        """Generate all datasets for both tables and save in combined files"""
        # Generate Table 2 datasets
        print("Generating Table 2 datasets...")
        table2_data = self.generate_table2_datasets(methods, item_counts, common_items, examinee_counts, seeds)
        
        # Generate Table 3 datasets (if parameters provided)
        table3_data = None
        if multi_pop_params:
            print("Generating Table 3 datasets...")
            table3_data = self.generate_table3_datasets(multi_pop_params, seeds)
        
        # Save the combined datasets
        if table2_data:
            self.save_combined_dataset(table2_data, self.table2_path, "table2")
        
        if table3_data:
            self.save_combined_dataset(table3_data, self.table3_path, "table3")
            
    def generate_table2_datasets(self, methods, item_counts, common_items, examinee_counts, seeds):
        """Generate all datasets for Table 2"""
        configs = []
        datasets = {}
        
        # Generate all configuration combinations
        for method, item_count, common_item, examinee_count, seed in itertools.product(
                methods, item_counts, common_items, examinee_counts, seeds):
            config = {
                'method': method,
                'item_count': item_count,
                'common_items': common_item,
                'examinee_count': examinee_count,
                'seed': seed,
                'multi_pop': False,
                'num_tests': 10,
                'table': 'table2'
            }
            config_id = self.generate_config_id(config)
            
            # Check if this config already exists in metadata
            if 'config_id' in self.metadata.columns and any((self.metadata['config_id'] == config_id).values):
                print(f"Dataset {config_id} already exists. Skipping.")
                continue
                
            configs.append((config_id, config))
        
        # Generate datasets
        total_configs = len(configs)
        print(f"Total Table 2 configurations to generate: {total_configs}")
        
        for i, (config_id, config) in enumerate(configs):
            print(f"Generating dataset {i+1}/{total_configs}: {config_id}")
            dgp = self.generate_single_dataset(config)
            
            # Store dataset in dictionary
            datasets[config_id] = self.extract_dgp_data(dgp, config)
        
        return datasets
    
    def generate_table3_datasets(self, multi_pop_params, seeds):
        """Generate all datasets for Table 3"""
        configs = []
        datasets = {}
        
        # Generate all Table 3 configurations
        for config_pair, common_item, examinee_count, seed in itertools.product(
                multi_pop_params['configurations'], 
                multi_pop_params['common_items'],
                multi_pop_params['examinee_counts'], 
                seeds):
            config = {
                'method': 'random',  # Multi-pop uses random assignment
                'item_count': 50,    # Fixed at 50 items per test 
                'common_items': common_item,
                'examinee_count': examinee_count,
                'seed': seed,
                'multi_pop': True,
                'mu_list': config_pair['mu_list'],
                'sigma2': config_pair['sigma2'],
                'num_tests': 2,     # Always 2 tests for multi-pop
                'table': 'table3'
            }
            config_id = self.generate_config_id(config)
            
            # Check if this config already exists in metadata
            if 'config_id' in self.metadata.columns and any((self.metadata['config_id'] == config_id).values):
                print(f"Dataset {config_id} already exists. Skipping.")
                continue
                
            configs.append((config_id, config))
        
        # Generate datasets
        total_configs = len(configs)
        print(f"Total Table 3 configurations to generate: {total_configs}")
        
        for i, (config_id, config) in enumerate(configs):
            print(f"Generating dataset {i+1}/{total_configs}: {config_id}")
            dgp = self.generate_single_dataset(config)
            
            # Store dataset in dictionary
            datasets[config_id] = self.extract_dgp_data(dgp, config)
        
        return datasets
    
    def extract_dgp_data(self, dgp, config):
        """Extract data from DGP instance for storage"""
        return {
            'theta': dgp.theta,
            'a': dgp.a,
            'b': dgp.b,
            'u': dgp.u,
            'item_ids': dgp.item_ids,
            'systematic': dgp.systematic,
            'assignments': dgp.assignments,
            'config': config,
            'total_items': dgp.total_items,
            'K': dgp.K,
            'J': dgp.J,
            'N': dgp.N,
            'common': dgp.common,
            'summary': dgp.summary()
        }
            
    def generate_single_dataset(self, config):
        """Generate a single dataset based on configuration"""
        systematic = config['method'] == 'systematic'
        
        # Use the correct number of tests from config
        num_tests = config.get('num_tests', 10)
        
        # Create DGP instance
        dgp = DGP2PLM(
            num_items=config['item_count'],
            num_examinees=config['examinee_count'],
            num_tests=num_tests,
            common_items=config['common_items'],
            systematic=systematic,
            seed=config['seed']
        )
        
        # Generate parameters and responses
        if config.get('multi_pop', False):
            dgp.generate_multi_population(
                mu_list=config['mu_list'],
                sigma2=config['sigma2'],
                seed=config['seed']
            )
        else:
            dgp.generate_parameters()
            
        dgp.simulate_responses()
        
        return dgp
        
    def save_combined_dataset(self, datasets, filepath, table_name):
        """Save all datasets for a table into a single file and update metadata"""
        print(f"Saving {len(datasets)} datasets to {filepath}")
        
        # Save the entire dictionary as a pickle file
        with open(filepath, 'wb') as f:
            pickle.dump(datasets, f)
        
        # Update metadata for each configuration
        new_rows = []
        for config_id, dataset in datasets.items():
            config = dataset['config']
            summary = dataset['summary']
            
            new_row = {
                'config_id': config_id,
                'table': table_name,  # Ensure table is always set correctly
                'method': config['method'],
                'item_count': config['item_count'],
                'common_items': config['common_items'],
                'examinee_count': config['examinee_count'],
                'seed': config['seed'],
                'multi_pop': config.get('multi_pop', False),
                'creation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'file_path': filepath,
                'total_items': summary['total_items'],
                'total_examinees': summary['total_examinees'],
                'num_tests': config.get('num_tests', 10)
            }
            
            # Add multi-pop specific params if applicable
            if config.get('multi_pop', False):
                new_row['mu_list'] = str(config['mu_list'])
                new_row['sigma2'] = config['sigma2']
            else:
                new_row['mu_list'] = None
                new_row['sigma2'] = None
                
            new_rows.append(new_row)
        
        if new_rows:
            # Create DataFrame for new rows
            new_df = pd.DataFrame(new_rows)
            
            if len(new_df) > 0:  # Only proceed if we actually have new data
                # Fix for FutureWarning: Create a copy of metadata to modify safely
                metadata_copy = self.metadata.copy()
                
                # Ensure columns exist in both dataframes with compatible types
                all_columns = set(list(metadata_copy.columns) + list(new_df.columns))
                
                # Create a new empty DataFrame with all columns to hold the result
                result_df = pd.DataFrame(columns=list(all_columns))
                
                # First append metadata rows (no warning this way)
                if len(metadata_copy) > 0:
                    result_df = pd.concat([result_df, metadata_copy], sort=False)
                
                # Then append new data rows
                result_df = pd.concat([result_df, new_df], sort=False)
                
                # Update metadata with result
                self.metadata = result_df.reset_index(drop=True)
                
                # Save updated metadata
                self.metadata.to_csv(self.metadata_path, index=False)
        
    def load_dataset(self, **filter_params):
        """Load dataset based on filter parameters and return a DGP2PLM instance"""
        if len(self.metadata) == 0:
            raise ValueError("Metadata is empty. No datasets available.")
        
        # Build query from filter parameters
        query_parts = []
        for key, value in filter_params.items():
            # Make sure the column exists before querying
            if key not in self.metadata.columns:
                raise ValueError(f"Column '{key}' not found in metadata. Available columns: {', '.join(self.metadata.columns)}")
                
            if isinstance(value, str):
                query_parts.append(f"{key} == '{value}'")
            else:
                query_parts.append(f"{key} == {value}")
        
        query = " & ".join(query_parts)
        filtered = self.metadata.query(query)
        
        if len(filtered) == 0:
            raise ValueError(f"No dataset matching criteria: {filter_params}")
        
        # Get the first match
        row = filtered.iloc[0]
        
        # Load the combined file containing this dataset
        with open(row['file_path'], 'rb') as f:
            combined_data = pickle.load(f)
        
        # Extract the specific dataset using its config_id
        dataset = combined_data[row['config_id']]
        
        # Reconstruct a DGP2PLM instance for compatibility with visualization code
        dgp = self.create_dgp_from_dataset(dataset)
        
        return dgp, row.to_dict()
    
    def create_dgp_from_dataset(self, dataset):
        """Recreate a DGP2PLM instance from a loaded dataset dictionary"""
        # Extract configuration
        config = dataset['config']
        
        # Create empty DGP instance
        dgp = DGP2PLM(
            num_items=config['item_count'],
            num_examinees=config['examinee_count'],
            num_tests=config['num_tests'],
            common_items=config['common_items'],
            systematic=config['method'] == 'systematic',
            seed=config['seed']
        )
        
        # Populate with loaded data
        dgp.theta = dataset['theta']
        dgp.a = dataset['a']
        dgp.b = dataset['b']
        dgp.u = dataset['u']
        dgp.item_ids = dataset['item_ids']
        dgp.systematic = dataset['systematic']
        dgp.assignments = dataset['assignments'] 
        dgp.total_items = dataset['total_items']
        dgp.K = dataset['K']
        dgp.J = dataset['J']
        dgp.N = dataset['N']
        dgp.common = dataset['common']
        
        return dgp
    
    def get_dataset_counts(self):
        """Get counts of datasets by table with error handling"""
        counts = {'table2': 0, 'table3': 0, 'unknown': 0}
        
        # Check if metadata exists and has the table column
        if len(self.metadata) == 0:
            return counts
        
        if 'table' not in self.metadata.columns:
            # Try to infer from file paths
            if 'file_path' in self.metadata.columns:
                counts['table2'] = sum(self.metadata['file_path'].str.contains('table2_combined', na=False))
                counts['table3'] = sum(self.metadata['file_path'].str.contains('table3_combined', na=False))
                counts['unknown'] = len(self.metadata) - counts['table2'] - counts['table3']
            else:
                counts['unknown'] = len(self.metadata)
        else:
            # Count by table column
            table_counts = self.metadata['table'].value_counts().to_dict()
            counts.update(table_counts)
        
        return counts
    
    def list_available_datasets(self, table=None):
        """List all available datasets with their configurations"""
        if len(self.metadata) == 0:
            print("No datasets available.")
            return pd.DataFrame()
        
        # Filter by table if specified
        if table:
            if 'table' in self.metadata.columns:
                filtered_metadata = self.metadata[self.metadata['table'] == table]
            else:
                # Fallback to file path filtering
                filtered_metadata = self.metadata[self.metadata['file_path'].str.contains(f'{table}_combined', na=False)]
        else:
            filtered_metadata = self.metadata
        
        # Select relevant columns for display
        display_cols = ['config_id', 'table', 'method', 'item_count', 'common_items', 
                       'examinee_count', 'num_tests', 'seed', 'multi_pop']
        
        # Add multi-pop specific columns if they exist
        if 'mu_list' in filtered_metadata.columns:
            display_cols.extend(['mu_list', 'sigma2'])
        
        # Only include columns that exist
        available_cols = [col for col in display_cols if col in filtered_metadata.columns]
        
        return filtered_metadata[available_cols].sort_values(['table', 'method', 'item_count'])
    
    def load_datasets_batch(self, filters_list):
        """Load multiple datasets based on a list of filter dictionaries"""
        datasets = []
        metadata_list = []
        
        for filters in filters_list:
            try:
                dgp, metadata = self.load_dataset(**filters)
                datasets.append(dgp)
                metadata_list.append(metadata)
                print(f"Loaded: {metadata['config_id']}")
            except ValueError as e:
                print(f"Failed to load dataset with filters {filters}: {e}")
                
        return datasets, metadata_list
    
    def load_table2_sample(self, method='random', item_count=20, common_items=5, 
                          examinee_count=500, seed=1):
        """Load a sample Table 2 dataset with default parameters"""
        return self.load_dataset(
            table='table2',
            method=method,
            item_count=item_count,
            common_items=common_items,
            examinee_count=examinee_count,
            seed=seed
        )
    
    def load_table3_sample(self, common_items=10, examinee_count=500, seed=1):
        """Load a sample Table 3 dataset with default parameters"""
        # Find first available Table 3 configuration
        table3_data = self.metadata[self.metadata['table'] == 'table3']
        if len(table3_data) == 0:
            raise ValueError("No Table 3 datasets available")
        
        # Get first match with specified parameters
        filtered = table3_data[
            (table3_data['common_items'] == common_items) & 
            (table3_data['examinee_count'] == examinee_count) & 
            (table3_data['seed'] == seed)
        ]
        
        if len(filtered) == 0:
            # Return first available if exact match not found
            row = table3_data.iloc[0]
            print(f"Exact match not found, loading: {row['config_id']}")
            return self.load_dataset(config_id=row['config_id'])
        else:
            row = filtered.iloc[0]
            return self.load_dataset(config_id=row['config_id'])
