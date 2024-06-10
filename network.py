import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.utils.weight_norm as weightNorm

import torchvision


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find("Conv2d") != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find("Linear") != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)


# Define the stochatic encoder model (map input X (one-hot code, flattern) into random gaussian variable)
class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim):
        super(Encoder, self).__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        self.shared_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.mean_layer = nn.Linear(hidden_dim, latent_dim)
        # scalar variance
        self.logvar_layer = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        encoded = self.shared_encoder(x)
        mu = self.mean_layer(encoded)
        logvar = self.logvar_layer(encoded)
        return mu, logvar


# Define the stochastic decoder model (by assuming the softmax distribution)
class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim, class_dim):
        super(Decoder, self).__init__()

        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        # number of attribute
        self.output_dim = output_dim
        # range of attribute
        self.class_dim = class_dim

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim * class_dim)
            # here the output probability should be faltterned vector
        )

    def forward(self, z):
        decoded = self.decoder(z)
        # reshape by 3D tensor， with a reconstructed model
        # thus we have 3D probability
        # actually we assume the conditional indepdence, the probability could be easily marginalized
        decoded = decoded.view(-1, self.output_dim, self.class_dim)
        reconstructed = torch.softmax(decoded, dim=2)
        return reconstructed



# =====================================================================================================
# Define a determinstic predictor to predict the bias value, this is similar to the Q learning strategy
# =====================================================================================================
def pred_init(args):
    pred_dict = {
        "mlp_reg": Predictor_MLP_REG(
            args.input_dim, args.pred_hidden_dim, args.pred_fea_dim
        ),
        "mlp_reg_cls": Predictor_MLP_REG_CLS(
            args.input_dim, args.pred_hidden_dim, args.pred_fea_dim, args.pred_cls_num
        ),
        "tf_reg": Predictor_TF_REG(
            args.input_dim, args.pred_hidden_dim, args.pred_fea_dim
        ),
        "tf_reg_cls": Predictor_TF_REG_CLS(
            args.input_dim, args.pred_hidden_dim, args.pred_fea_dim, args.pred_cls_num
        ),
    }
    return pred_dict[args.pred_type]


class Predictor_MLP_REG(nn.Module):
    """
    two-layer regression
    """

    def __init__(self, input_dim, hidden_dim, pred_fea_dim):
        super(Predictor_MLP_REG, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.pred_fea_dim = pred_fea_dim

        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, pred_fea_dim),
            nn.ReLU(),
        )
        self.reg = nn.Linear(pred_fea_dim, 1)

        # init
        self.model.apply(init_weights)
        self.reg.apply(init_weights)

    def forward(self, x):
        fea = self.model(x)
        bias_value = self.reg(fea)
        # bias_value = F.relu(self.linear(fea))

        return bias_value.squeeze(), None, fea


class Predictor_MLP_REG_CLS(nn.Module):
    """
    two-layer regression
    """

    def __init__(self, input_dim, hidden_dim, pred_fea_dim, pred_cls_num):
        super(Predictor_MLP_REG_CLS, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.pred_fea_dim = pred_fea_dim
        self.pred_cls_num = pred_cls_num

        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, pred_fea_dim),
            nn.ReLU(),
            # nn.Softmax()
        )
        self.reg_fea = nn.Sequential(nn.Linear(pred_fea_dim, pred_fea_dim), nn.ReLU())
        self.reg = nn.Linear(pred_fea_dim, 1)
        self.cls_fea = nn.Linear(pred_fea_dim, pred_fea_dim)
        self.cls = weightNorm(nn.Linear(pred_fea_dim, pred_cls_num), name="weight")

        # init
        self.model.apply(init_weights)
        self.reg_fea.apply(init_weights)
        self.reg.apply(init_weights)
        self.cls_fea.apply(init_weights)
        self.cls.apply(init_weights)

    def forward(self, x, return_fea=False):
        x = self.model(x)
        fea_reg = self.reg_fea(x)
        fea_cls = self.cls_fea(x)
        fea_reg = x + fea_reg
        fea_cls = x + fea_cls

        bias_value = self.reg(fea_reg)
        cls_out = self.cls(fea_cls)

        return bias_value.squeeze(), cls_out, fea_cls


# ===============================================================================================================================================
# refer to: https://github.com/Evergreen0929/Kaggle_House_Prices_Transformer_Pytorch/blob/main/Kaggle_House_Prices_PyTorch/python_code/model.py
# ===============================================================================================================================================
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        self.q = nn.Linear(in_features, in_features)
        self.k = nn.Linear(in_features, in_features)
        self.v = nn.Linear(in_features, in_features)
        self.fc1 = nn.Linear(in_features, hidden_features)

        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x0 = x
        q = self.q(x).unsqueeze(2)
        k = self.k(x).unsqueeze(2)
        v = self.v(x).unsqueeze(2)
        attn = q @ k.transpose(-2, -1)
        # print(attn.size())
        attn = attn.softmax(dim=-1)
        x = (attn @ v).squeeze(2)
        x = x0 + x
        x1 = x
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = x1 + x
        return x


class Predictor_TF_REG(nn.Module):
    def __init__(self, in_features, hidden_features, pred_fea_dim, drop=0.0):
        super().__init__()
        self.Block1 = Mlp(
            in_features=hidden_features,
            hidden_features=hidden_features,
            act_layer=nn.GELU,
            drop=drop,
        )
        self.Block2 = Mlp(
            in_features=hidden_features,
            hidden_features=hidden_features,
            act_layer=nn.GELU,
            drop=drop,
        )
        self.linear0 = nn.Linear(in_features, hidden_features)
        self.linear_fea = nn.Linear(hidden_features, pred_fea_dim)
        self.reg = nn.Linear(pred_fea_dim, 1)

        # init
        self.Block1.apply(init_weights)
        self.Block2.apply(init_weights)
        self.linear0.apply(init_weights)
        self.linear_fea.apply(init_weights)
        self.reg.apply(init_weights)

    def forward(self, x):
        input = self.Block2(self.Block1(self.linear0(x)))
        fea = self.linear_fea(input)
        out = self.reg(fea)

        return out.squeeze(), None, fea


class Predictor_TF_REG_CLS(nn.Module):
    def __init__(
        self, in_features, hidden_features, pred_fea_dim, pred_cls_num, drop=0.0
    ):
        super().__init__()
        self.Block1 = Mlp(
            in_features=hidden_features,
            hidden_features=hidden_features,
            act_layer=nn.GELU,
            drop=drop,
        )
        self.Block2 = Mlp(
            in_features=hidden_features,
            hidden_features=hidden_features,
            act_layer=nn.GELU,
            drop=drop,
        )
        self.linear0 = nn.Linear(in_features, hidden_features)
        self.linear_fea = nn.Linear(hidden_features, pred_fea_dim)

        self.reg_fea = nn.Sequential(nn.Linear(pred_fea_dim, pred_fea_dim), nn.ReLU())
        self.reg = nn.Linear(pred_fea_dim, 1)
        self.cls_fea = nn.Linear(pred_fea_dim, pred_fea_dim)
        self.cls = weightNorm(nn.Linear(pred_fea_dim, pred_cls_num), name="weight")

        # init
        self.Block1.apply(init_weights)
        self.Block2.apply(init_weights)
        self.linear0.apply(init_weights)
        self.linear_fea.apply(init_weights)
        self.reg_fea.apply(init_weights)
        self.reg.apply(init_weights)
        self.cls_fea.apply(init_weights)
        self.cls.apply(init_weights)

    def forward(self, x):
        input = self.Block2(self.Block1(self.linear0(x)))
        fea = self.linear_fea(input)
        fea_reg = self.reg_fea(fea)
        fea_cls = self.cls_fea(fea)
        fea_reg = fea + fea_reg
        fea_cls = fea + fea_cls

        bias_value = self.reg(fea_reg)
        cls_out = self.cls(fea_cls)

        return bias_value.squeeze(), cls_out, fea_cls


# =============================================================
# network for celebA
# =============================================================


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class ResNet18_Encoder(nn.Module):
    def __init__(self, pretrained):
        super().__init__()
        if pretrained:
            self.resnet = torchvision.models.resnet18(pretrained=True)
        else:
            self.resnet = torchvision.models.resnet18()
        self.resnet.fc = Identity()
        self.resnet.avgpool = Identity()

    def forward(self, x):
        outputs = self.resnet(x)
        return outputs.view(-1, 512, 8, 8)


class LinearModel(nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 1)
        self.relu = nn.ReLU()
        self.avg = nn.AdaptiveAvgPool2d(output_size=(1, 1))

    def forward(self, x):
        x = self.avg(x).view(-1, 512)
        x = self.fc1(x)
        x = self.relu(x)
        outputs = self.fc2(x)
        return torch.sigmoid(outputs.squeeze())

    def fea_1d(self, x):
        return self.avg(x).view(-1, 512)


class F_Model_TOXIC(nn.Module):
    def __init__(self):
        super(F_Model_TOXIC, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(768, 400),
            nn.ELU(),
            nn.Linear(400, 200),
            nn.ELU(),
            nn.Linear(200, 100),
            nn.ELU(),
            nn.Linear(100, 1),
        )
    
    def forward(self, x):
        x = self.model(x)
        x = torch.sigmoid(x.squeeze(dim=-1))
        return x


# =============================================================
# network for GAN/WGAN
# =============================================================
class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)

        return validity