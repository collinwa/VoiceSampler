from VoiceMixer import MixVoiceData
from processing import *
from u_net import *
import torch
import numpy as np

EPOCHS = 5

if __name__ == '__main__':
    u_net = UNet(depth=3)

    if torch.cuda.is_available():
        u_net.cuda()

    optimizer = torch.optim.Adam(u_net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    data_gen = MixVoiceData('gdrive/My Drive/mp3_instrumental/','gdrive/My Drive/mp3_singers/')
    for epoch in range(EPOCHS):
        for idx, (x, y) in enumerate(data_gen.generate_data()):
            # skip training for samples that are too short
            if (x is None) or (y is None):
                continue

            x = np.expand_dims(x, axis=0)
            y = np.expand_dims(y, axis=0)
            x = torch.tensor(x).float()

            x = torch.transpose(x, 1, 2)
            y = torch.transpose(y, 1, 2)

            if torch.cuda.is_available():
                x = x.to(device="cuda")
                y = y.to(device="cuda")
            
            y_pred = u_net(x)

            loss = compute_mse(y_pred, y)
            print('epoch: %i training_step: %i current_loss: %f' % (epoch, idx, loss))
            
            loss.backward() # compute gradients
            optimizer.step() # update weights
            optimizer.zero_grad() # zero-out the gradients for the next iteration
