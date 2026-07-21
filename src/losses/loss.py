import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedMAELoss(nn.Module):
    """
    Masked Mean Absolute Error loss for NDVI forecasting.

    prediction:
        [B, T_future, 1]

    target:
        [B, T_future, 1]

    mask:
        [B, T_future]

    """


    def __init__(self):
        super().__init__()


    def forward(
        self,
        prediction,
        target,
        mask
    ):

        # Absolute error
        loss = torch.abs(
            prediction - target
        )

        # [B,T,1]
        mask = mask.unsqueeze(-1).float()

        # remove padded timestep
        loss = loss * mask


        # average only valid targets
        loss = loss.sum() / mask.sum()


        return loss


class BaselineDynamicsMAELoss(nn.Module):
    """
    Same masked-MAE interface as MaskedMAELoss, but decomposes the error
    into two parts so within-sample temporal dynamics aren't drowned out
    by cross-sample baseline differences:

        baseline  = masked mean of the sequence over the future window
        dynamics  = sequence - baseline (the ups/downs around it)

    Plain masked MAE lets baseline error (large, because different AOIs
    have very different NDVI levels) dominate the gradient, so the model
    learns "what level is this sample at" but not "how does it move
    within that window" -- which is exactly what the low within-sample
    correlation showed.

    Drop-in replacement: same __call__ signature as MaskedMAELoss.

    The dynamics term uses Huber loss instead of MAE: dynamics targets are
    per-timestep deviations from the sample's own baseline, computed from
    Whittaker-smoothed NDVI that still carries residual cloud/atmosphere
    noise at individual dates. Huber is quadratic (like MSE) for small
    errors -- giving a well-behaved gradient near zero -- and linear (like
    MAE) for large errors, so a handful of noisy points can't dominate the
    gradient the way they would under MSE. The baseline term stays MAE:
    it's a per-sample mean over the whole window, so noise mostly averages
    out and doesn't need the extra robustness.

    Args:
        lambda_dynamics: weight on the dynamics term. Start at 3.0.
            Raise it (5.0, 8.0...) if within-sample correlation is still
            low after training; lower it if baseline MAE gets noticeably
            worse. Tune against validation within-sample correlation,
            not just MAE/RMSE.
        lambda_baseline: weight on the baseline term, usually leave at 1.0.
        dynamics_loss_type: "huber" (default) or "mae". Huber trades exact
            gradient scale for robustness to residual noise in the
            Whittaker-smoothed targets; "mae" recovers the original
            behavior for comparison.
        huber_delta: only used when dynamics_loss_type="huber". Transition
            point between quadratic and linear regions, in NDVI units.
            Errors below this are penalized like MSE, above it like MAE.
            Default 0.05 -- roughly the scale of a "normal" within-window
            NDVI fluctuation; larger deviations are more likely residual
            noise than real signal.
    """

    def __init__(
        self,
        lambda_dynamics: float = 3.0,
        lambda_baseline: float = 1.0,
        dynamics_loss_type: str = "huber",
        huber_delta: float = 0.05,
    ):
        super().__init__()
        assert dynamics_loss_type in ("huber","mae")
        self.lambda_dynamics = lambda_dynamics
        self.lambda_baseline = lambda_baseline
        self.dynamics_loss_type = dynamics_loss_type
        self.huber_delta = huber_delta

    def _compute(self, prediction, target, mask):
        # prediction, target: [B, T, 1] -> squeeze to [B, T]
        pred = prediction.squeeze(-1)
        tgt = target.squeeze(-1)

        mask_f = mask.float()  # [B, T]
        valid_counts = mask_f.sum(dim=1, keepdim=True).clamp(min=1)  # [B, 1]

        # ---- baseline: masked mean over the future window, per sample ----
        pred_baseline = (pred * mask_f).sum(dim=1, keepdim=True) / valid_counts    # [B, 1]
        target_baseline = (tgt * mask_f).sum(dim=1, keepdim=True) / valid_counts    # [B, 1]

        baseline_loss = torch.abs(pred_baseline - target_baseline).mean()

        # ---- dynamics: deviation from each sample's own baseline ----
        pred_dynamics = (pred - pred_baseline) * mask_f
        target_dynamics = (tgt - target_baseline) * mask_f

        # padded positions are 0-0=0 under both losses, so this masking
        # trick (zero out, then divide by valid count) still holds
        if self.dynamics_loss_type == "huber":
            dynamics_err = F.huber_loss(
                pred_dynamics, target_dynamics, reduction="none", delta=self.huber_delta
            )
        else:
            dynamics_err = torch.abs(pred_dynamics - target_dynamics)

        dynamics_loss = dynamics_err.sum() / mask_f.sum().clamp(min=1)

        total_loss = (
            self.lambda_baseline * baseline_loss
            + self.lambda_dynamics * dynamics_loss
        )

        return total_loss, baseline_loss, dynamics_loss

    def forward(
        self,
        prediction,
        target,
        mask
    ):
        total_loss, _, _ = self._compute(prediction, target, mask)
        return total_loss

    def forward_with_components(self, prediction, target, mask):
        """Same as forward(), but also returns the individual loss terms
        for logging during training (e.g. to watch dynamics_loss trend
        down over epochs)."""
        total_loss, baseline_loss, dynamics_loss = self._compute(prediction, target, mask)

        return total_loss, {
            "baseline_loss": baseline_loss.item(),
            "dynamics_loss": dynamics_loss.item(),
            "total_loss": total_loss.item(),
        }


if __name__ == "__main__":

    torch.manual_seed(0)

    prediction = torch.randn(4, 8, 1)
    target = torch.randn(4, 8, 1)
    mask = torch.ones(4, 8)
    mask[:, -2:] = 0  # simulate padded timesteps

    # ---- original loss still works unchanged ----
    plain = MaskedMAELoss()
    print("Plain MaskedMAE loss:", plain(prediction, target, mask).item())

    # ---- new decomposed loss, same call signature ----
    combo = BaselineDynamicsMAELoss(lambda_dynamics=3.0)
    print("BaselineDynamics loss:", combo(prediction, target, mask).item())

    total, components = combo.forward_with_components(prediction, target, mask)
    print("Components:", components)

    # sanity check: perfect prediction -> loss should be ~0
    perfect_pred = target.clone()
    perfect_loss = combo(perfect_pred, target, mask)
    print("\nPerfect prediction loss (should be ~0):", perfect_loss.item())
    assert perfect_loss.item() < 1e-6
    print("OK")