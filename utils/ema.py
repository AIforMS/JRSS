from copy import deepcopy

import torch


class ModelEMA(object):
    """
    ema_model = ModelEMA(device, model, ema_decay)
    get model:
      model_to_test = ema_model.ema
    """
    def __init__(self, device, model, decay):
        self.ema = deepcopy(model)
        self.ema.to(device)
        self.ema.eval()
        self.decay = decay
        self.ema_has_module = hasattr(self.ema, 'module')
        # Fix EMA. https://github.com/valencebond/FixMatch_pytorch thank you!
        self.param_keys = [k for k, _ in self.ema.named_parameters()]
        self.buffer_keys = [k for k, _ in self.ema.named_buffers()]
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        """
        EMA using in fixmatch.
        :param model: teacher model
        """
        needs_module = hasattr(model, 'module') and not self.ema_has_module
        with torch.no_grad():
            msd = model.state_dict()
            esd = self.ema.state_dict()
            for k in self.param_keys:
                if needs_module:
                    j = 'module.' + k
                else:
                    j = k
                model_v = msd[j].detach()
                ema_v = esd[k]
                esd[k].copy_(ema_v * self.decay + (1. - self.decay) * model_v)

            for k in self.buffer_keys:
                if needs_module:
                    j = 'module.' + k
                else:
                    j = k
                esd[k].copy_(msd[j])

    def update_UAMT(self, global_step, model):
        """
        EMA using in UAMT.
        :param global_step: training step, use the true average until the exponential average is more correct
        :param model: teacher model
        """
        alpha = min(1 - 1 / (global_step + 1), self.decay)
        for ema_param, param in zip(self.ema.parameters(), model.parameters()):
            ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


if __name__ == '__main__':
    """
    test wether EMA FixMatch is the same with UAMT?
    The ansr is YES!!!
    """
    from models import U_Network as UNet

    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')
    ema_decay = 0.99

    vol_size = [160, 192, 160]
    nf_enc = [16, 32, 32, 32]
    nf_dec = [32, 32, 32, 32, 8, 8]       # vm1
    # nf_dec = [32, 32, 32, 32, 32, 16, 16] # vm2

    model = UNet(len(vol_size), nf_enc, nf_dec, bn=True).to(device)
    ema_model_1 = ModelEMA(device, model, ema_decay)
    ema_model_2 = ModelEMA(device, model, ema_decay)

    ema_model_1.update(model=model)
    ema_model_2.update_UAMT(global_step=1000, model=model)

    for ema_1_param, ema_2_param in zip(ema_model_1.ema.parameters(), ema_model_2.ema.parameters()):
        assert ema_1_param.equal(ema_2_param), "EMA FixMatch is not the same with UAMT!"
