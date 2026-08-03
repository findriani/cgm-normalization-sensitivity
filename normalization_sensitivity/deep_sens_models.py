"""
deep_sens_models.py -- the published simple mid-fusion backbone, branch-selectable
=================================================================================
E2 needs the SAME deep architecture under three modality configurations (`cgm`,
`static_all`, `cgm_static`) and two CGM normalization arms. This module rebuilds the
published `build_current_multimodal` backbone with the branches made selectable.

WHAT IS IDENTICAL TO THE PUBLISHED MODEL
----------------------------------------
Every layer, width, activation, dropout rate, optimizer and loss is copied from
`lightdl_models.build_current_multimodal`:

    CGM branch     Bidirectional(LSTM(32)) -> Dense(32, relu)
    static branch  Dense(32, relu) -> Dense(16, relu)
    fusion head    Dense(32, relu) -> Dropout(0.3) -> three linear heads

`_heads` and `_compile` are IMPORTED from lightdl_models rather than reimplemented, so
the output ordering [out_30, out_60, out_120], the per-output MSE losses, the loss
weights and the Adam learning rate cannot drift from the published configuration.

WHAT IS DELIBERATELY DIFFERENT: NO DYNAMIC BRANCH
--------------------------------------------------
The published model also carries a GRU(16) branch over the HR/activity channels. E2 drops
it, for two reasons that must both survive into the manuscript:

  1. NO E2 CONFIGURATION USES THE DYNAMIC CHANNELS. `cgm`, `static_all` and `cgm_static`
     are CGM and/or static only -- exactly as in E1 and the reference ablation. A branch
     fed only by unused channels would add parameters and optimization noise while
     carrying no information.
  2. The occlusion and ablation results already show these channels contribute nothing;
     the activity/HR question is answered elsewhere and is not what E2 is testing.

So E2's backbone is the published simple mid-fusion RESTRICTED to the modalities under
test. It is not a reproduction of the published model in full, and the text must not say
it is. What matters for E2's validity is that the backbone is IDENTICAL ACROSS ARMS --
the only thing that varies between A0 and A1 is the CGM normalization.

PARAMETER COUNTS
----------------
`count_params` is reported per configuration and recorded in the manifest. Unmatched
parameter counts were a specific reviewer complaint about the original fusion claim; here
the two arms share a configuration's architecture exactly, so counts are equal by
construction -- but they are still reported rather than asserted informally.
=================================================================================
"""
from tensorflow.keras import layers, models

# Imported so the head layout, losses, loss weights and optimizer cannot drift from the
# published configuration. Private by name, deliberate by intent -- see module docstring.
from lightdl_models import _heads, _compile


def build_midfusion(timesteps, n_static, use_cgm=True, use_static=True,
                    aux=False, lr=1e-3):
    """Published mid-fusion backbone with selectable branches (no dynamic branch).

    At least one branch must be enabled. With a single branch the concatenation is
    skipped and the branch output feeds the fusion head directly, which is what
    `Concatenate` on one input would do anyway -- written explicitly so the graph has no
    degenerate one-input merge."""
    if not (use_cgm or use_static):
        raise ValueError("at least one of use_cgm / use_static must be True")

    inputs, branches = [], []

    if use_cgm:
        ci = layers.Input(shape=(timesteps, 1), name="cgm_input")
        c = layers.Bidirectional(layers.LSTM(32), name="cgm_bilstm32")(ci)
        c = layers.Dense(32, activation="relu", name="cgm_dense32")(c)
        inputs.append(ci)
        branches.append(c)

    if use_static:
        si = layers.Input(shape=(n_static,), name="static_input")
        s = layers.Dense(32, activation="relu", name="stat_dense32")(si)
        s = layers.Dense(16, activation="relu", name="stat_dense16")(s)
        inputs.append(si)
        branches.append(s)

    fused = branches[0] if len(branches) == 1 else \
        layers.Concatenate(name="fusion_concat")(branches)
    h = layers.Dense(32, activation="relu", name="fusion_dense32")(fused)
    h = layers.Dropout(0.3, name="fusion_dropout")(h)

    model = models.Model(inputs, _heads(h, aux), name="midfusion")
    return _compile(model, aux, lr)


# Which branches each E2 configuration enables. `persistence_5min` fits nothing.
BRANCHES = {
    "cgm":        {"use_cgm": True,  "use_static": False},
    "static_all": {"use_cgm": False, "use_static": True},
    "cgm_static": {"use_cgm": True,  "use_static": True},
}


def model_inputs(cfg, X, S):
    """Assemble the input list for one configuration, in the builder's input order."""
    spec = BRANCHES[cfg]
    out = []
    if spec["use_cgm"]:
        out.append(X[:, :, 0:1])          # CGM channel only; dynamic channels unused
    if spec["use_static"]:
        out.append(S)
    return out


if __name__ == "__main__":
    for cfg, spec in BRANCHES.items():
        m = build_midfusion(12, 17, **spec)
        print(f"{cfg:12s} inputs={[i.shape[1:] for i in m.inputs]} "
              f"params={m.count_params()}")
