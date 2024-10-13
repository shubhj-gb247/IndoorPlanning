import h5py
import torch
from torch.utils.data import Dataset


class FloorPlanDataset(Dataset):
    def test_func():
        return 5


    def __init__(self, hf_files):
        """
        Args:
        hf_files (list): list of paths to hdf5 files containing the data
        """

        self.hf_files = hf_files
        self.datasets = []
        self.indices = []

        # Load the datasets and build index mapping

        cummulative_samples = 0
        for file_idx, hf_file in enumerate(self.hf_files):
            hf = h5py.File(hf_file, "r")
            num_samples = hf["site_dim"].shape[0]
            self.datasets.append(hf)
            self.indices.extend([(file_idx, i) for i in range(num_samples)])
            cummulative_samples += num_samples

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        file_idx, image_idx = self.indices[
            index
        ]  # will return a tuple denoting (batch_idx , image_idx)
        hf = self.datasets[file_idx]

        # input data

        # site_dim
        site_dim = hf["site_dim"][image_idx]
        site_dim = torch.tensor(site_dim, dtype=torch.float32)


        # room categories
        room_category_len = hf["room_category_lens"][image_idx]
        room_category_padded = hf["room_category"][image_idx]
        room_category = room_category_padded[:room_category_len]
        room_category = torch.tensor(room_category, dtype=torch.long)

        # target data
        # test from here
        # room areas
        room_area_len = hf["room_area_norm_lens"][image_idx]
        room_area_padded = hf["room_area_norm"][image_idx]
        room_area_norm = room_area_padded[:room_area_len]
        room_area_norm = torch.tensor(room_area_norm, dtype=torch.float32)

        # Room Bounding Boxes
        room_bb_len = hf["room_bb_norm_lens"][image_idx]
        room_bb_padded = hf["room_bb_norm"][image_idx]
        room_bb_norm = room_bb_padded[:room_bb_len]
        room_bb_norm = torch.tensor(room_bb_norm, dtype=torch.float32)

        # Edge Data
        edge_start = hf["edge_offsets"][image_idx]
        edge_end = hf["edge_offsets"][image_idx + 1]
        edges_src = hf["edges_src"][edge_start:edge_end]
        edges_tgt = hf["edges_tgt"][edge_start:edge_end]
        edge_index = torch.stack(
            [
                torch.tensor(edges_src, dtype=torch.long),
                torch.tensor(edges_tgt, dtype=torch.long),
            ],
            dim=0,
        )
        edge_door = torch.tensor(
            hf["edges_door"][edge_start:edge_end], dtype=torch.float32
        )
        edge_location = torch.tensor(
            hf["edges_location"][edge_start:edge_end], dtype=torch.float32
        )

        # Prepare data dictionaries
        input_data = {
            "site_dim": site_dim,
            "room_category": room_category,
        }

        target_data = {
            "room_area_norm": room_area_norm,
            "room_bb_norm": room_bb_norm,
            "edge_index": edge_index,
            "edge_door": edge_door,
            "edge_location": edge_location,
        }

        return input_data, target_data
