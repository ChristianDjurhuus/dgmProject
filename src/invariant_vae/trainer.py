import torch
import torch.nn.functional as F
import pytorch_lightning as pl
#from invariant_vae import VAE
from invariant_vaeV2 import VAE
from invariant_vaeV2 import VariationalInference
from torch.distributions import Bernoulli
import matplotlib.pyplot as plt
import numpy as np


class Trainer(pl.LightningModule):

    def __init__(self, emb_dim, hidden_dim):
        super().__init__()
        self.vae = VAE(emb_dim, hidden_dim)
        self.vi = VariationalInference()

    def forward(self, x):
        #y, _, mu, log_var = self.model(x)
        #return y, mu, log_var
        #loss, diagnostics, outputs = self.vi(self.vae, x)
        return self.vi(self.vae, x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.bernoulli()
        #y, mu, log_var = self(x)
        #loss, _, _ = self.loss(y.squeeze(), x.squeeze(), mu, log_var)
        loss, diagnostics, outputs = self(x)

        #self.log("loss", loss)
        self.log("elbo", diagnostics["elbo"].mean())
        self.log("log_px", diagnostics["log_px"].mean())
        self.log("kld", diagnostics["kld"].mean())
        
        logs={"train_loss": loss}

        
        batch_dictionary={
            #REQUIRED: It ie required for us to return "loss"
            "loss": loss,
             
            #optional for batch logging purposes
            "log": logs,
 
        }
 
        return batch_dictionary

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.bernoulli()
        #x = x.round()
        #y, mu, log_var = self(x)
        #loss, _, _ = self.loss(y.squeeze(), x.squeeze(), mu, log_var)
        #self.log("val_loss", loss)
        loss, diagnostics, outputs = self(x)
        #self.log("val_loss", loss)
        self.log("val_elbo", diagnostics["elbo"].mean())
        self.log("val_log_px", diagnostics["log_px"].mean())
        self.log("val_kld", diagnostics["kld"].mean())
        
        if batch_idx == 0 and self.current_epoch%10==0:
            torch.save(self.vae.state_dict(), f'trained_modelsv2/invariant_vae_{self.current_epoch}.pth')
            nrows = 5
            ncols = 2
            fig, axs = plt.subplots(nrows, 2 * ncols, figsize=(15,15))
            for i in range(nrows):
                for j in range(ncols):
                    idx = np.random.choice(x.shape[0], replace=False)
                    # Original and reconstruction
                    x_ = x.squeeze().cpu().detach().numpy()
                    x_hat = outputs['px'].mean.squeeze().cpu().detach().numpy()

                    axs[i, 2 * j].imshow(x_[idx,:,:], cmap='gray')
                    axs[i, 2 * j].set_title(f"Original image ({y[idx]})")
                    axs[i, 2 * j].axis('off')

                    axs[i, 2 * j + 1].imshow(x_hat[idx,:,:], cmap='gray')
                    axs[i, 2 * j + 1].set_title(f"Reconstructed image ({y[idx]})")
                    axs[i, 2 * j + 1].axis('off')
                
            plt.savefig(f"results_epoch{self.current_epoch}.png")
            plt.close()
           


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.vae.parameters(), lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=optimizer,
            gamma=0.995,
        )
        scheduler = {
            'scheduler': lr_scheduler,
            'interval': 'epoch',
            'frequency': 2
        }
        return [optimizer]#, [scheduler]
    
    #def loss(self, y, x, mu, log_var):#, batch):
        '''
        Computes ELBO loss (Cross entropy + KLD)
        https://ai.stackexchange.com/questions/27341/in-variational-autoencoders-why-do-people-use-mse-for-the-loss
        '''
    #    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - (log_var).exp())
        #KLD /= (batch * 32) #TODO: generalise this part

        #loss = torch.nn.BCEWithLogitsLoss(reduce='None')
        #BCE = loss(x_hat, x)

        #return BCE + KLD, BCE, KLD

    #    mseloss = torch.nn.MSELoss(reduce='Mean')
    #    MSE = mseloss(y, x)
    #    return MSE + KLD, MSE, KLD

    def training_epoch_end(self, outputs):
        
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
 
        # logging using tensorboard logger
        self.logger.experiment.add_scalar("Loss/Train",
                                            avg_loss,
                                            self.current_epoch)
         

 
        epoch_dictionary={
            # required
            'loss': avg_loss}
