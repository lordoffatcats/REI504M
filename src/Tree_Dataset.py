import json
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import torchvision.transforms.functional as TF
import json
import re
from pathlib import Path


def parse_vector(s: str, dtype=torch.float32) -> torch.Tensor:
    """ Utilty function for TreesDataset
    Parse string of form:
        'X=0.080 Y=-1.538 Z=-0.113'
    into:
        tensor([0.080, -1.538, -0.113])
    """
    # values = re.findall(r"[XYZ]=([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", s)
    values = [float(part.split("=")[1]) for part in s.split()]

    if len(values) != 3:
        raise ValueError(f"Could not parse 3D vector: {s}")

    return torch.tensor(values, dtype=dtype)

# TODO: add the option to normalize coordinates of trees/camera based on training set statistics.
# TODO: cache numeric data into Torch tensors in init function and only load images in get_item.
class TreesDataset(Dataset):
    """
    Dataset where each item corresponds to one tree.

    Input:
        Trees.json and Images folder

    Output item:
        {
            X: {
                "views": {
                    "images": Tensor[V, 3, x_pixels, y_pixels] (3 for R,G,B)
                    "camera_location": Tensor[V, 3],
                    "rotation": Tensor[V, 4],
                    "intrinsics": Tensor[V, 4],
                    "projection_matrix": Tensor[V, 4, 4],
                },
                "ground_level": Tensor[1],
            },

            y: {
                "circles": {
                    "centre": Tensor[C, 3],
                    "normal": Tensor[C, 3],
                    "radius": Tensor[C],
                },
                "dbh": Tensor[1],
            }
        }

    C = number of circles for this tree (can vary between trees)
    V = number of views (líklega 7)
    """

    def __init__(
        self,
        json_path,
        image_root=None,
        normalize_to_ground=True,
        dtype=torch.float32,
    ):
        """ Creates a dataset from json.
        Input:
            json_path: path of dataset in json format
            image_root: path of images folder referenced in json file
            normalize_to_ground: whether to subtract ground level from camera and circle coordinates
        """
        self.json_path = Path(json_path)
        self.normalize_to_ground = normalize_to_ground
        self.image_root = (
            Path(image_root)
            if image_root is not None
            else self.json_path.parent
        )
        self.dtype = dtype

        with open(self.json_path, "r") as f:
            data = json.load(f)

        self.elements = data["Elements"]

    def __len__(self):
        return len(self.elements)

    def load_image(self, image_path):
        """ Loads an image from path, converts to RBG and returns tensor
        """
        image_path = Path(image_path)
        if not image_path.is_absolute():
            image_path = self.image_root / image_path
        image = Image.open(image_path).convert("RGB")
        return TF.to_tensor(image)

    def __getitem__(self, idx):
        tree = self.elements[idx]
        tree_id = idx # unique id_for this tree
        
        # simple features
        ground_level = float(tree["GroundLevel"])
        dbh = float(tree["DBH"])

        # TODO: perhaps add species as One-Hot encoded vector (or embedding)
        

        # ------- Circles -------
        circle_centres = []
        circle_normals = []
        circle_radii = []

        for circle in tree["Circles"]: # the number of circles may be different between trees
            centre = parse_vector(circle["Centre"]) # ------- circle X, Y, Z position vector -------
            normal = parse_vector(circle["Normal"]) # ------- circle X, Y, Z normal vector -------
            radius = float(circle["Radius"]) # ------- circle radius -------
            if self.normalize_to_ground: # ------- normalization to ground level -------
                # Assuming Y is the vertical axis.
                centre[1] -= ground_level
            circle_centres.append(centre)
            circle_normals.append(normal)
            circle_radii.append(radius)
        circles = { 
            "centre": torch.stack(circle_centres),
            "normal": torch.stack(circle_normals),
            "radius": torch.tensor(
                circle_radii,
                dtype=self.dtype,
            ),
        }

        # ------- Views -------
        images = []
        camera_locations = []
        rotations = []
        intrinsics = []
        projection_matrices = []

        for view in tree["Views"]:
            # ------- load images -------
            image = self.load_image(view["Image"])
            images.append(image)

            # ------- camera X, Y, Z vector -------
            camera_location = parse_vector(view["CameraLocation"])
            if self.normalize_to_ground: # ------- normalization to ground level -------
                # Assuming Y is vertical.
                camera_location[1] -= ground_level
            camera_locations.append(camera_location)

            # ------- rotation quaternions -------
            rotation = torch.tensor(
                [
                    float(view["RotationQuaternionW"]),
                    float(view["RotationQuaternionX"]),
                    float(view["RotationQuaternionY"]),
                    float(view["RotationQuaternionZ"]),
                ],
                dtype=self.dtype,
            )
            rotations.append(rotation)

            # ------- focal length and principal point -------
            intrinsics.append(
                torch.tensor(
                    [
                        float(view["fx"]),
                        float(view["fy"]),
                        float(view["cx"]),
                        float(view["cy"]),
                    ],
                    dtype=self.dtype,
                )
            )

        """
            # ------- camera projection matrix ------- (can probably be ignored methinks)
            projection_matrices.append(
                torch.tensor(
                    [
                        [
                            float(view["M11"]),
                            float(view["M12"]),
                            float(view["M13"]),
                            float(view["M14"]),
                        ],
                        [
                            float(view["M21"]),
                            float(view["M22"]),
                            float(view["M23"]),
                            float(view["M24"]),
                        ],
                        [
                            float(view["M31"]),
                            float(view["M32"]),
                            float(view["M33"]),
                            float(view["M34"]),
                        ],
                        [
                            float(view["M41"]),
                            float(view["M42"]),
                            float(view["M43"]),
                            float(view["M44"]),
                        ],
                    ],
                    dtype=self.dtype,
                )
            )
        """

        # ------- stacking into nested data structure -------
        views = {
            "images": torch.stack(images),
            "camera_location": torch.stack(camera_locations),
            "rotation": torch.stack(rotations),
            "intrinsics": torch.stack(intrinsics),
            #"projection_matrix": torch.stack(projection_matrices),
        }

        # ------- X values ------- (for now)
        X = {
            "views": views,
            "ground_level": torch.tensor(
                ground_level,
                dtype=self.dtype,
            ),
        }

        # ------- y labels ------- (for now)
        y = {
            "dbh": torch.tensor(dbh, dtype=self.dtype),
            "circles": circles,
        }

        return tree_id, X, y

def collate_trees(batch):
    """ Collate function that allows each tree to have a different number of circles.
    """
    tree_ids, Xs, ys = zip(*batch)

    return (
        torch.tensor(tree_ids, dtype=torch.long),
        {
            "views": {
                key: torch.stack([X["views"][key] for X in Xs])
                for key in Xs[0]["views"]
            },
            "ground_level": torch.stack(
                [X["ground_level"] for X in Xs]
            ),
        },
        {
            "circles": {
                key: [y["circles"][key] for y in ys]
                for key in ys[0]["circles"]
            },
            "dbh": torch.stack([y["dbh"] for y in ys]),
        },
    )

def get_dataloaders(dataset, batch_size=32, val_ratio=0.2, test_ratio=0.2, seed=42, num_workers=0, pin_memory=False, persistent_workers=False):
    """ Creates train,val,test, dataloaders for a given input dataset.
    Input:
        dataset: an instance of TreesDataset,
        batch_size: size of training batch,
        val_ratio: ratio of examples in validation set,
        test_ratio: ratio of examples in test set,
        seed: seed so the same dataset split can be reproduced,
        num_workers: specifies how many parallel subprocesses are used for loading data.
            num_workers=0: means that the main training process handles all loading (may cause bottleneck).
            num_workers>0: means that multiple parallel subprocesses can load data and store it in a queue (yields better GPU utilization).
            According to Google a good starting point is 2 workers, some say that matching the amount of CPU cores is also a good heuristic.
        The ratio of training examples is inferred from val_ratio and test_ratio.
        pin_memory: Page locks in ram so CUDA driver can use direct memory access. 
        persistent_workers: Keeps dataloading workers alive between epochs instead of shutting down and restarting them.
    Returns:
        train_loader, val_loader, test_loader: DataLoader instances for the dataset splits.
    """
    dataset_size=len(dataset)

    assert (val_ratio >= 0 and test_ratio >= 0 and val_ratio+test_ratio <= 1), "Invalid val/test ratios."
    val_size = int(val_ratio * dataset_size)
    test_size = int(test_ratio * dataset_size)
    train_size = dataset_size-val_size-test_size

    generator = torch.Generator().manual_seed(seed)

    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=generator,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_trees,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_trees,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_trees,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader
