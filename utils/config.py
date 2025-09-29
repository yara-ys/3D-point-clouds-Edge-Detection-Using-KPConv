
import json

class Config:
    def __init__(self):
    
        self.root_path  = "/content/drive/MyDrive/Challenge-ABC"
        self.train_path = self.root_path + "/Train"
        self.val_path   = self.root_path + "/Validation"

        # ---- Data / features
        self.use_normals = True
        self.use_ssm     = True
        self.normalize_ssm = True
        self.num_classes = 2
        self.conv_radius = 2.5 
        
        # ---- Multiscale geometry 
        self.first_subsampling_dl = 0.5
        self.layer_multipliers    = [1, 2, 4]           # 3 levels
        self.max_neighbors        = [12, 16, 16]       # per level 
        self.base_radius_factor   = 2.5                    # auto base radius
        self.layer_radii          = [self.conv_radius * self.first_subsampling_dl * float(m) for m in self.layer_multipliers]   #set list of floats to lock radii


        #Pooling
        self.pool_cap = 8   # children per coarse (shadow padded) 

        #KPConv kernel
        self.num_kernel_points = 15
        self.KP_inward_scale   = 0.85
        self.influence_mode    = "gaussian" #linear or gaussian

        ###### Architecture widths 
        self.widths = [48, 64, 96] 

        ###### Decoder upsampling
        self.upsample_mode = "nearest"                     #can be nearest or interp
        self.interp_k      = 3                             
        self.interp_mode   = "idw"                         #can be idw or gaussian
        self.interp_sigma  = None                         

        ##### Training
        self.batch_size    = 2
        self.max_epochs    = 60
        self.learning_rate = 1e-3
        self.weight_decay  = 1e-4
        self.optimizer     = "AdamW"
        self.lr_schedule   = "cosine"
        self.warmup_steps  = 0
        self.grad_clip     = 0.0
        self.amp           = True
        self.class_weights = [1.0, 6.0]
        self.seed   = 42
        self.device = "cuda"
        self.kp_chunkK = 4 #new
        self.use_residual = True
        self.use_dice  = True  
        self.default_threshold = 0.25
        
    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.__dict__, f, indent=2)

    @classmethod
    def load(cls, path):
        with open(path, "r") as f:
            data = json.load(f)
        cfg = cls()
        cfg.__dict__.update(data)
        return cfg