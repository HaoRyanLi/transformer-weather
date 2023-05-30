"""
=====
Author: Hao L.
=====
"""
import logging
import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple
from .embedding_model import EmbeddingModel, EmbeddingTrainingHead
from trphysx.config.configuration_phys import PhysConfig
from torch.autograd import Variable

logger = logging.getLogger(__name__)
# Custom types
Tensor = torch.Tensor
TensorTuple = Tuple[torch.Tensor]
FloatTuple = Tuple[float]

class ERA5Embedding(EmbeddingModel):
    """Embedding Koopman model for the 2D flow around a era5 data

    Args:
        config (PhysConfig): Configuration class with transformer/embedding parameters
    """
    model_name = "embedding_era5"

    def __init__(self, config: PhysConfig) -> None:
        """Constructor method
        """
        super().__init__(config)

        # Encoder conv. net
        self.observableNet = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=(5, 5), stride=3, padding=2, padding_mode='circular'),
            # nn.BatchNorm2d(16),
            nn.ReLU(True),
            # 16, 240, 480
            nn.Conv2d(16, 32, kernel_size=(5, 5), stride=3, padding=2, padding_mode='circular'),
            # nn.BatchNorm2d(32),
            nn.ReLU(True),
            # 32, 80, 160
            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=2, padding=1, padding_mode='circular'),
            # nn.BatchNorm2d(64),
            nn.ReLU(True),
            # 64, 40, 80
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=2, padding=1, padding_mode='circular'),
            # nn.BatchNorm2d(128),
            nn.ReLU(True),
            # 128, 20, 40
            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=2, padding=1, padding_mode='circular'),
            # nn.BatchNorm2d(256),
            nn.ReLU(True),
            # 256, 10, 20
            nn.Conv2d(256, 64, kernel_size=(3, 3), stride=1, padding=1, padding_mode='circular'),
            # 64, 10, 20
        )

        self.observableNetFC = nn.Sequential(
            nn.Linear(64*10*20, 8*10*20),
            nn.LeakyReLU(0.02, inplace=True),
            nn.Linear(8*10*20, config.n_embd),
            nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon),
            # nn.BatchNorm1d(config.n_embd, eps=config.layer_norm_epsilon),
            # nn.Dropout(config.embd_pdrop)
        )

        self.recoveryNetFC = nn.Sequential(
            nn.Linear(config.n_embd, 8*10*20),
            nn.LeakyReLU(1.0, inplace=True),
            nn.Linear(8*10*20, 64*10*20),
            nn.LeakyReLU(0.02, inplace=True),
        )

        # Decoder conv. net
        self.recoveryNet = nn.Sequential(
            # 64, 10, 20
            nn.Conv2d(64, 256, kernel_size=(3, 3), stride=1, padding=1, padding_mode='circular'),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.02, inplace=True),
            # 256, 10, 20
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(256, 128, kernel_size=(3, 3), stride=1, padding=1, padding_mode='circular'),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.02, inplace=True),
            # 128, 20, 40
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(128, 64, kernel_size=(3, 3), stride=1, padding=1, padding_mode='circular'),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.02, inplace=True),
            # 64, 40, 80
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, 32, kernel_size=(3, 3), stride=1, padding=1, padding_mode='circular'),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.02, inplace=True),
            # 32, 80, 160
            nn.Upsample(scale_factor=3, mode='bilinear', align_corners=True),
            nn.Conv2d(32, 16, kernel_size=(5, 5), stride=1, padding=2, padding_mode='circular'),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.02, inplace=True),
            # 16, 240, 480
            nn.Upsample(scale_factor=3, mode='bilinear', align_corners=True),
            nn.Conv2d(16, 16, kernel_size=(5, 5), stride=1, padding=2, padding_mode='circular'),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.02, inplace=True),
            # 16, 720, 1440
            nn.Conv2d(16, 3, kernel_size=(3, 3), stride=1, padding=1, padding_mode='circular'),
        )


        # Learned Koopman operator
        self.kMatrixDiag = nn.Parameter(torch.ones(config.n_embd))

        # Off-diagonal indices
        xidx = []
        yidx = []
        for i in range(1, 10):
            yidx.append(np.arange(i, self.config.n_embd))
            xidx.append(np.arange(0, self.config.n_embd - i))

        self.xidx = torch.LongTensor(np.concatenate(xidx))
        self.yidx = torch.LongTensor(np.concatenate(yidx))
        self.kMatrixUT = nn.Parameter(0.01 * torch.rand(self.xidx.size(0)))
        self.kMatrixLT = nn.Parameter(0.01 * torch.rand(self.xidx.size(0)))

        # Normalization occurs inside the model
        self.register_buffer('mu', torch.tensor(0.))
        self.register_buffer('std', torch.tensor(1.))
        logger.info('Number of embedding parameters: {}'.format(super().num_parameters))

    def forward(self, x: Tensor, extra_var: Tensor) -> TensorTuple:
        """Forward pass

        Args:
            x (Tensor): [B, 3, H, W] Input feature tensor
            extra_var (Tensor): [Extra, H, W] extra variables of the era5 data in the mini-batch

        Returns:
            (TensorTuple): Tuple containing:

                | (Tensor): [B, config.n_embd] Koopman observables
                | (Tensor): [B, 3, H, W] Recovered feature tensor
        """
        # Concat extra variables as extra feature maps.
        x = torch.cat([x, extra_var * torch.ones_like(x[:,:1])], dim=1)

        x = self._normalize(x)
        g0 = self.observableNet(x)
        g = self.observableNetFC(g0.view(g0.size(0),-1))

        # Decode
        out0 = self.recoveryNetFC(g).view(-1, 64, 10, 20)
        out = self.recoveryNet(out0)
        xhat = self._unnormalize(out)
        return g, xhat

    def embed(self, x: Tensor, extra_var: Tensor) -> Tensor:
        """Embeds tensor of state variables to Koopman observables

        Args:
            x (Tensor): [B, 3, H, W] Input feature tensor
            extra_var (Tensor): [Extra, H, W] extra variables of the era5 data in the mini-batch

        Returns:
            (Tensor): [B, config.n_embd] Koopman observables
        """
        # Concat viscosities as a feature map
        x = torch.cat([x, extra_var * torch.ones_like(x[:,:1])], dim=1)
        x = self._normalize(x)
        g = self.observableNet(x)
        g = self.observableNetFC(g.view(g.size(0), -1))

        return g

    def recover(self, g: Tensor) -> Tensor:
        """Recovers feature tensor from Koopman observables

        Args:
            g (Tensor): [B, config.n_embd] Koopman observables

        Returns:
            (Tensor): [B, 2, H, W] Physical feature tensor
        """
        out = self.recoveryNetFC(g).view(-1, 64, 10, 20)
        out = self.recoveryNet(out)
        x = self._unnormalize(out)
        return x

    def koopmanOperation(self, g: Tensor) -> Tensor:
        """Applies the learned Koopman operator on the given observables

        Args:
            g (Tensor): [B, config.n_embd] Koopman observables

        Returns:
            (Tensor): [B, config.n_embd] Koopman observables at the next time-step
        """
        # Koopman operator
        kMatrix = Variable(torch.zeros(g.size(0), self.config.n_embd, self.config.n_embd)).to(self.devices[0])
        
        # Populate the off diagonal terms
        kMatrix[:, self.xidx, self.yidx] = self.kMatrixUT
        kMatrix[:, self.yidx, self.xidx] = self.kMatrixLT
        
        # Populate the diagonal
        ind = np.diag_indices(kMatrix.shape[1])
        kMatrix[:, ind[0], ind[1]] = self.kMatrixDiag

        # Apply Koopman operation
        gnext = torch.bmm(kMatrix, g.unsqueeze(-1))
        self.kMatrix = kMatrix

        return gnext.squeeze(-1) # Squeeze empty dim from bmm


    @property
    def koopmanOperator(self, requires_grad: bool =True) -> Tensor:
        """Current Koopman operator

        Args:
            requires_grad (bool, optional): If to return with gradient storage. Defaults to True

        Returns:
            Tensor: Full Koopman operator tensor
        """
        if not requires_grad:
            return self.kMatrix.detach()
        else:
            return self.kMatrix

    @property
    def koopmanDiag(self):
        return self.kMatrixDiag

    def _normalize(self, x):
        x = (x - self.mu.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)) / self.std.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        return x

    def _unnormalize(self, x):
        return self.std.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * x + self.mu.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)


class ERA5EmbeddingTrainer(EmbeddingTrainingHead):
    """Training head for the ERA5 data embedding model

    Args:
        config (PhysConfig): Configuration class with transformer/embedding parameters
    """
    def __init__(self, config: PhysConfig) -> None:
        """Constructor method
        """
        super().__init__()
        self.embedding_model = ERA5Embedding(config)

    def forward(self, states: Tensor, extra_var: Tensor) -> FloatTuple:
        """Trains model for a single epoch

        Args:
            states (Tensor): [B, T, 3, H, W] Time-series feature tensor
            extra_var (Tensor): [Extra, H, W] extra variables of the era5 data in the mini-batch

        Returns:
            FloatTuple: Tuple containing:
            
                | (float): Koopman based loss of current epoch
                | (float): Reconstruction loss
        """

        self.embedding_model.train()
        device = self.embedding_model.devices[0]

        loss_reconstruct = 0
        mseLoss = nn.MSELoss()

        xin0 = states[:,0].to(device) # Initial time-step
        extra_var = extra_var.to(device)

        # Model forward for initial time-step
        g0, xRec0 = self.embedding_model(xin0, extra_var)
        loss = (1e1)*mseLoss(xin0, xRec0)
        loss_reconstruct = loss_reconstruct + mseLoss(xin0, xRec0).detach()

        g1_old = g0
        # Loop through time-series
        for t0 in range(1, states.shape[1]):
            xin0 = states[:,t0,:].to(device) # Next time-step
            _, xRec1 = self.embedding_model(xin0, extra_var)
            # Apply Koopman transform
            g1Pred = self.embedding_model.koopmanOperation(g1_old)
            xgRec1 = self.embedding_model.recover(g1Pred)

            # Loss function
            loss = loss + (1e1)*mseLoss(xgRec1, xin0) + (1e1)*mseLoss(xRec1, xin0) \
                + (1e-2)*torch.sum(torch.pow(self.embedding_model.koopmanOperator, 2))
                
            loss_reconstruct = loss_reconstruct + mseLoss(xRec1, xin0).detach()
            g1_old = g1Pred

        return loss, loss_reconstruct

    def evaluate(self, states: Tensor, extra_var: Tensor) -> Tuple[float, Tensor, Tensor]:
        """Evaluates the embedding models reconstruction error and returns its
        predictions.

        Args:
            states (Tensor): [B, T, 3, H, W] Time-series feature tensor
            extra_var (Tensor): [Extra, H, W] extra variables of the era5 data in the mini-batch

        Returns:
            Tuple[Float, Tensor, Tensor]: Test error, Predicted states, Target states
        """
        self.embedding_model.eval()
        device = self.embedding_model.devices[0]

        mseLoss = nn.MSELoss()

        # Pull out targets from prediction dataset
        yTarget = states[:,1:].to(device)
        xInput = states[:,:-1].to(device)
        yPred = torch.zeros(yTarget.size()).to(device)
        extra_var = extra_var.to(device)

        # Test accuracy of one time-step
        for i in range(xInput.size(1)):
            xInput0 = xInput[:,i].to(device)
            g0 = self.embedding_model.embed(xInput0, extra_var)
            g0 = self.embedding_model.koopmanOperation(g0)
            yPred0 = self.embedding_model.recover(g0)
            yPred[:,i] = yPred0.squeeze().detach()

        test_loss = mseLoss(yTarget, yPred)
        plain_error = mseLoss(yTarget, xInput)

        return test_loss, plain_error, yPred, yTarget