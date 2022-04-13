import os
import os.path
from typing import Callable, Optional
from torchvision.datasets import ImageFolder
from torchvision import transforms


class UNDAImageFolder(ImageFolder):
    def __init__(
            self,
            root: str,
            mode: str,
            is_valid_file: Optional[Callable[[str], bool]] = None,
    ):
        with open(f'{root}_{mode}.txt', 'r') as f:
            info = f.readlines()
            self.valid_files = set(list(map(lambda x: x.split(' ')[0].split('/')[-1], info)))

        super().__init__(
            root=root,
            is_valid_file=is_valid_file,
            transform=transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
        )

    def find_classes(self, directory: str):
        """ Finds the class folders in a dataset (skips empty dirs).

            See :class:`DatasetFolder` for details.
        """
        classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())

        def is_empty_unda_class(mark: str):
            path = os.path.join(self.root, mark)
            samples = [entry.name for entry in os.scandir(path) if entry.is_file() and entry.name in self.valid_files]
            return len(samples) == 0

        classes = list(filter(lambda x: not is_empty_unda_class(x), classes))
        if not classes:
            raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")

        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx
