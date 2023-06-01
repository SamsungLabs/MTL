from collections import defaultdict
import torch

from . import mtl_metrics


class BasicBalancer(torch.nn.Module):
    def __init__(self, compute_stats=False):
        super().__init__()
        self.compute_stats = compute_stats
        self.info = None
        self.losses = defaultdict(float)

    def set_losses(self, losses):
        self.losses = {task_id: float(losses[task_id]) for task_id in losses}

    def compute_metrics(self, G: torch.Tensor):
        self.info = mtl_metrics.compute_metrics(G)

    def add_model_parameters(self, model):
        pass

    @staticmethod
    def zero_grad_model(model):
        model.zero_grad()

    @staticmethod
    def apply_decoder_scaling(decoders, weights):
        for i, decoder in enumerate(decoders.values()):
            for p in decoder.parameters():
                if p.grad is not None:
                    p.grad.mul_(weights[i])

    @staticmethod
    def scale_task_specific_params(task_specific_params: dict, weights: dict):
        for task_id in task_specific_params:
            for p in task_specific_params[task_id]:
                if p.grad is not None:
                    p.grad.mul_(weights[task_id])

    @staticmethod
    def set_encoder_grad(encoder, grad_vec):
        offset = 0
        for p in encoder.parameters():
            if p.grad is None:
                continue
            _offset = offset + p.grad.shape.numel()
            p.grad.data = grad_vec[offset:_offset].view_as(p.grad)
            offset = _offset

    @staticmethod
    def set_shared_grad(shared_params, grad_vec):
        offset = 0
        for p in shared_params:
            if p.grad is None:
                continue
            _offset = offset + p.grad.shape.numel()
            p.grad.data = grad_vec[offset:_offset].view_as(p.grad)
            offset = _offset

    @staticmethod
    def get_G_wrt_shared(losses, shared_params, update_decoder_grads=False):
        grads = []
        for task_id in losses:
            cur_loss = losses[task_id]
            if not update_decoder_grads:
                grad = torch.cat([p.flatten() if p is not None else torch.zeros_like(shared_params[i]).flatten()
                                  for i, p in enumerate(torch.autograd.grad(cur_loss, shared_params,
                                                               retain_graph=True, allow_unused=True))])
            else:
                for p in shared_params:
                    if p.grad is not None:
                        p.grad.data.zero_()

                cur_loss.backward(retain_graph=True)
                grad = torch.cat([p.grad.flatten().clone() if p.grad is not None else torch.zeros_like(p).flatten()
                                  for p in shared_params])

            grads.append(grad)

        for p in shared_params:
            if p.grad is not None:
                p.grad.data.zero_()

        return torch.stack(grads, dim=0)

    @staticmethod
    def get_model_G_wrt_shared(hrepr, targets, encoder, decoders, criteria, loss_fn=None,
                               update_decoder_grads=False, return_losses=False):
        if loss_fn is None:
            loss_fn = lambda task_task_id: criteria[task_task_id](decoders[task_task_id](hrepr), targets[task_task_id])

        grads = []
        losses = {}
        for task_id in criteria:
            cur_loss = loss_fn(task_id)
            if not update_decoder_grads:
                grad = torch.cat([p.flatten()
                                  for p in torch.autograd.grad(cur_loss, encoder.parameters(),
                                                               retain_graph=True, allow_unused=True)
                                  if p is not None])
            else:
                encoder.zero_grad()
                cur_loss.backward(retain_graph=True)
                grad = torch.cat([p.grad.flatten().clone() for p in encoder.parameters() if p.grad is not None])

            grads.append(grad)
            losses[task_id] = cur_loss

        grads = torch.stack(grads, dim=0)
        if return_losses:
            return grads, losses
        else:
            return grads

    @staticmethod
    def get_model_G_wrt_hrepr(hrepr, targets, model, criteria, loss_fn=None,
                              update_decoder_grads=False, return_losses=False):

        _hrepr = hrepr.data.detach().clone().requires_grad_(True)
        if loss_fn is None:
            loss_fn = lambda task_task_id: criteria[task_task_id](model.decoders[task_task_id](_hrepr),
                                                                  targets[task_task_id])

        grads = []
        losses = {}
        for task_id in criteria:
            cur_loss = loss_fn(task_id)
            if not update_decoder_grads:
                grad = torch.cat([p.flatten()
                                  for p in torch.autograd.grad(cur_loss, _hrepr,
                                                               retain_graph=False, allow_unused=True)
                                  if p is not None])
            else:
                if _hrepr.grad is not None:
                    _hrepr.grad.data.zero_()
                cur_loss.backward(retain_graph=False)
                grad = _hrepr.grad.flatten().clone()

            grads.append(grad)
            losses[task_id] = cur_loss

        grads = torch.stack(grads, dim=0)
        if return_losses:
            return grads, losses
        else:
            return grads

    @staticmethod
    def compute_losses(data: torch.Tensor, targets: dict, model: torch.nn.Module, criteria: dict, **kwargs):
        BasicBalancer.zero_grad_model(model)
        hrepr = model.encoder(data)

        losses = {}
        for task_id in criteria:
            losses[task_id] = criteria[task_id](model.decoders[task_id](hrepr), targets[task_id])
        return losses, hrepr

    def step_with_model(self, data: torch.Tensor, targets: dict, model: torch.nn.Module, criteria: dict,
                        **kwargs) -> None:
        losses, hrepr = self.compute_losses(data, targets, model, criteria)
        self.step(losses=losses,
                  shared_params=list(model.encoder.parameters()),
                  task_specific_params={task_id: list(model.decoders.parameters()) for task_id in model.decoders},
                  shared_representation=hrepr,
                  last_shared_layer_params=list(model.last_shared_layer.parameters())
                                          if model.last_shared_layer is not None
                                          else None)

    def step(self, losses, shared_params, task_specific_params, shared_representation=None,
             last_shared_layer_params=None) -> None:
        raise NotImplementedError("Balancer requires model to be specified. "
                                  "Use 'step_with_model' method for this balancer")
