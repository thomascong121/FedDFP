import h5py
import torch
from PIL import Image
from torchvision import transforms
import openslide
import numpy as np
import pandas as pd
import random
import os
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


class Camelyon16Raw(Dataset):
    def __init__(
        self, X_dtype=torch.float32,
            y_dtype=torch.float32,
            debug=False,
            data_path=None,
            logger=None,
            feature_type='R50_features',
            use_stain=False,
            top_k=-1,
            feature_abandon=False,
    ):
        self.data_path = data_path
        self.labels = pd.read_csv('/scratch/iq24/cc0395/FedDDHist/data/cam16/labels.csv')
        self.metadata = pd.read_csv('/scratch/iq24/cc0395/FedDDHist/data/cam16/metadata.csv')
        self.X_dtype = X_dtype
        self.y_dtype = y_dtype
        self.debug = debug
        if feature_type == 'Heter':
            extractor_pool_1024 = ['R50_features', 'UNI_features']
            extractor_pool_512 = ['PLIP_features', 'CONCH_features']
            seed = 10
            random.seed(seed)
            if feature_abandon in extractor_pool_1024:
                extractor_pool_1024.remove(feature_abandon)
                ft_to_use = random.choice(extractor_pool_1024)
            elif feature_abandon in extractor_pool_512:
                extractor_pool_512.remove(feature_abandon)
                ft_to_use = random.choice(extractor_pool_512)
            elif not feature_abandon:
                # when use heterogenous features but have not decided to the feature type
                ft_to_use = random.choice(extractor_pool_1024 + extractor_pool_512)
            else:
                raise ValueError(f'Feature type {feature_abandon} not found')
            self.feature_type = ft_to_use
        else:
            self.feature_type = feature_type
        print(f'Using {self.feature_type} features')
        self.features_paths = []
        self.features_labels = []
        self.features_centers = []
        self.features_sets = []
        self.perms = {}
        self.top_k = top_k
        self.use_stain = use_stain
        npys_list = [
            e
            for e in sorted(self.labels['filenames'].tolist())
            if e.lower() not in ("normal_086.pt", "test_049.pt")
        ]
        random.seed(0)
        random.shuffle(npys_list)
        for slide in npys_list:
            slide_name = os.path.basename(slide).split(".")[0].lower()
            slide_id = int(slide_name.split("_")[1])
            slide_pt = slide_name + ".pt"
            label_from_metadata = int(
                self.metadata.loc[
                    [
                        e.split(".")[0] == slide_name
                        for e in self.metadata["slide_name"].tolist()
                    ],
                    "label",
                ].item()
            )
            center_from_metadata = int(
                self.metadata.loc[
                    [
                        e.split(".")[0] == slide_name
                        for e in self.metadata["slide_name"].tolist()
                    ],
                    "hospital_corrected",
                ].item()
            )
            label_from_data = int(self.labels.loc[self.labels['filenames']==slide.lower()].tumor)
            if label_from_data == 0:
                label_str = "normal"
            else:
                label_str = "tumor"

            if "test" not in str(slide).lower():
                if slide_name.startswith("normal"):
                    # Normal slide
                    if slide_id > 100:
                        center_label = 1
                    else:
                        center_label = 0
                    label_from_slide_name = 0  # Normal slide
                elif slide_name.startswith("tumor"):
                    # Tumor slide
                    if slide_id > 70:
                        center_label = 1
                    else:
                        center_label = 0
                    label_from_slide_name = 1  # Tumor slide
                assert label_from_slide_name == label_from_data, "This shouldn't happen"
                assert center_label == center_from_metadata, "This shouldn't happen"
                stage = "train"
            else:
                stage = "test"
            # 加载特征，特征路径是特征文件路径。
            slide_pt_pth = f'{self.data_path}/{stage}/{label_str}/{self.feature_type}/pt_files/{slide_pt}'
            if not os.path.exists(slide_pt_pth):
                # print(f"Warning: {slide_pt_pth} does not exist")
                if logger is not None:
                    logger.warning(f"Warning: {slide_pt_pth} does not exist")
                continue

            assert label_from_metadata == label_from_data
            self.features_paths.append(slide_pt_pth)
            self.features_labels.append(label_from_data)
            self.features_centers.append(center_from_metadata)
            self.features_sets.append(stage)
        if len(self.features_paths) < len(self.labels.index):
            if logger is not None:
                logger.warning(
                    "Warning you are operating on a reduced dataset in  DEBUG mode with"
                    f" in total {len(self.features_paths)}/{len(self.labels.index)}"
                    " features."
                )

    def __len__(self):
        return len(self.features_paths)

    def __getitem__(self, idx, path=False):
        start = 0
        slide_pth = self.features_paths[idx]
        slide_label = self.features_labels[idx]
        X = torch.load(slide_pth).to(self.X_dtype)
        #处理特征维度，如果特征维度小于2，则进行reshape。
        if len(X.size()) < 2:
            if 'ViT' in slide_pth:
                ft_len = 384
            elif 'PLIP' in slide_pth or 'CONCH' in slide_pth:
                ft_len = 512
            else:
                ft_len = 1024
            #将X变换为二维张量。
            X = X.view(-1, ft_len)
        y = torch.from_numpy(np.asarray(slide_label)).to(self.y_dtype)
        #是否增加颜色特征，好像现在暂时不需要。
        if self.use_stain:
            if 'R50' in slide_pth:
                stain_feature_pth = slide_pth.replace('R50_features', 'Color_label')
            elif 'ViT' in slide_pth:
                stain_feature_pth = slide_pth.replace('ViT_features', 'Color_label')
            else:
                stain_feature_pth = slide_pth.replace('pre_extracted_feature', 'Color_label')
            stain_features = torch.load(stain_feature_pth)
            X_color = torch.zeros((X.size()[0], X.size()[1] + 1))
            X_color[:, :-1] = X
            X_color[:, -1] = stain_features
            X = X_color
        #随机采样，如果self.top_k > 0，且特征张量的第一个维度大于self.top_k，则随机采样。
        if self.top_k > 0:
            if X.size(0) > self.top_k:
                idx = torch.randperm(X.size(0))[:self.top_k]
                X = X[idx]
        if path:
            return X, y, slide_pth
        #X是特征张量，y是标签张量。
        return X, y
        # if idx not in self.perms:
        #     self.perms[idx] = np.random.default_rng(42).permutation(X.shape[0])

        # return X, y, self.perms[idx]


class FedCamelyon16(Camelyon16Raw):
    def __init__(
        self,
        center: int = 0,
        train: bool = True,
        pooled: bool = False,
        X_dtype: torch.dtype = torch.float32,
        y_dtype: torch.dtype = torch.float32,
        debug: bool = False,
        data_path: str = None,
        logger=None,
        require_image: bool = False,
        image_size: int = 256,
        feature_type='R50_features',
        use_stain=False,
        top_k=-1,
        feature_abandon=False,
        deterministic=False,
        fold_idx: int = 0,
    ):
        """
        Cf class docstring
        """
        super().__init__(
            X_dtype=X_dtype,
            y_dtype=y_dtype,
            debug=debug,
            data_path=data_path,
            logger=logger,
            feature_type=feature_type,
            use_stain=use_stain,
            top_k=top_k,
            feature_abandon=feature_abandon,
        )
        assert center in [0, 1]
        self.centers = [center]
        if pooled:
            self.centers = [0, 1]
        if train:
            self.sets = ["train"]
        else:
            self.sets = ["test"]

        to_select = [
            (self.features_sets[idx] in self.sets)
            and (self.features_centers[idx] in self.centers)
            for idx, _ in enumerate(self.features_centers)
        ]
        self.features_paths = [
            fp for idx, fp in enumerate(self.features_paths) if to_select[idx]
        ]
        self.features_sets = [
            fp for idx, fp in enumerate(self.features_sets) if to_select[idx]
        ]
        self.features_labels = [
            fp for idx, fp in enumerate(self.features_labels) if to_select[idx]
        ]
        self.features_centers = [
            fp for idx, fp in enumerate(self.features_centers) if to_select[idx]
        ]
        #图像预处理过程，将图像转换为指定大小，并转换为张量。
        self.trsforms = []
        self.trsforms.append(transforms.Resize(image_size))
        self.trsforms.append(transforms.ToTensor())
        # self.trsforms.append(transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
        self.trsforms = transforms.Compose(self.trsforms)
        #是否需要根据类别获得索引。
        if require_image:
            self.indices_classes = self.get_idx_per_class()

    def get_idx_per_class(self):
        idx_per_class = {}
        for idx, label in tqdm(enumerate(self.features_labels)):
            if label not in idx_per_class:
                idx_per_class[label] = []
            slide_patches = torch.load(self.features_paths[idx])
            idx_per_class[label].append([idx, slide_patches.size(0)])
        return idx_per_class
     #从类别c中随机选取n_slide个幻灯片（slides)，并从幻灯片中选择n_patch个幻灯片。
    def get_image(self, c, n_slide, n_patch=0):
        # Get N-patches from N-slides of class c
        #是元组(slide_idx, n_patches)，表示幻灯片的索引和幻灯片中的幻灯片数量。
        idx_slide_n_patches = random.sample(self.indices_classes[c], n_slide)
        sample_slides = []
        for i in range(len(idx_slide_n_patches)):
            slide_idx, n_patches = idx_slide_n_patches[i]
            #slide_pt是通过torch.load加载的Pytorch张量，是一个二维张量，形状为[n_patches, 1024]
            #n_patches表示幻灯片中patch数量，feature_dim表示特征维度，这里是1024。
            slide_pt = torch.load(self.features_paths[slide_idx])
            slide_h5 = self.features_paths[slide_idx].replace("pt_files", "h5_files").replace(".pt", ".h5")
            slide_pth = self.features_paths[slide_idx].replace("CAMELYON16_patches", "CAMELYON16").replace(
                "R50_features/pt_files/", "").replace("pt", "tif")
            slide_patches = []
            if slide_pt.size(0) < n_patch or n_patch == 0:
                # load entire slide for real image sampling
                slide_patches.append(slide_pt)
            #如果采样补丁，则从幻灯片中随机选择n_patch个幻灯片。从HDF5文件中读取坐标，并使用OpenSlide读取图像。
            else:
                sampled_patch_idx = random.sample(range(n_patches), n_patch)
                for patch_idx in sampled_patch_idx:
                    with h5py.File(slide_h5, 'r') as hdf5_file:
                        #在.h5文件中读取图像块的坐标coord。
                        coord = hdf5_file['coords'][patch_idx]
                        wsi = openslide.open_slide(slide_pth)
                        img = wsi.read_region(coord, 0, (256, 256)).convert('RGB')
                        img = self.trsforms(img)
                        slide_patches.append(img.unsqueeze(0))
                del wsi
            sample_slides.append(torch.cat(slide_patches, dim=0).unsqueeze(0))
        #最后返回的张量形状为[n_slide, n_patches, 3, 256, 256]。
        return torch.cat(sample_slides, dim=0)

if __name__=='__main__':
    train_dict = {'UMCU': 0, 'RUMC': 0}
    test_dict = {'UMCU': 0, 'RUMC': 0}
    print('====================')
    # To load the first center as a pytorch dataset
    center0 = FedCamelyon16(center=0, train=True, data_path='/g/data/iq24/CAMELYON16_patches')
    # To load the second center as a pytorch dataset
    center1 = FedCamelyon16(center=1, train=True, data_path='/g/data/iq24/CAMELYON16_patches')
    train_dict['UMCU'] = len(center0)
    train_dict['RUMC'] = len(center1)

    center0 = FedCamelyon16(center=0, train=False, data_path='/g/data/iq24/CAMELYON16_patches')
    center1 = FedCamelyon16(center=1, train=False, data_path='/g/data/iq24/CAMELYON16_patches')
    test_dict['UMCU'] = len(center0)
    test_dict['RUMC'] = len(center1)
    print(train_dict)
    print(test_dict)

    X, y = iter(DataLoader(center0, batch_size=1, shuffle=True, num_workers=0)).next()
    print(X.size(), y.size())
