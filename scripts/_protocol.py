"""The constants the evaluation protocol is defined by.

They are kept here, and imported everywhere else, so that the systems whose
readings are averaged and the two thresholds the reading rests on have exactly
one definition: a value that appears in several files is a value that can
disagree with itself without anything failing.

The module imports nothing, so the emitters and the analysis scripts can read
it without pulling in torch, yaml or the package itself.

  BACKBONES      the three generic architectures, retrained by
                 scripts/phase3_audit_pad.py, which takes --backbone.
  AUDITED        the systems whose readings are averaged into one analysis
                 unit. CFD-PAD is one of them: it is trained by us, on the same
                 data and task, so it is a replicate of the same measurement
                 and not a separate category. It has its own entry point
                 (scripts/phase3_cfdpad.py) only because its loss and its
                 channel-importance pass do not fit the --backbone dispatch.
  P_THRESHOLD    p_{u,k} at or above which Phase 2 issues a directional
                 prediction on an anchor (main paper, Section V-C).
  NOISE_FLOOR    |Delta/sigma| below which a shift is not read as an effect
                 (main paper, Section IV-D). Placed at the first tenth above
                 every matched-size control, whose largest excursion is 1.16:
                 those withdraw the same quantity of training data without
                 withdrawing a material, so they bound what the reduction of
                 the training set alone produces. Section S5 of the
                 supplementary recomputes the reading over a range of both
                 values.
  short()        the abbreviations every table of the paper uses. They live
                 here for the same reason the constants do: five emitters
                 shortening the same names five times is five chances to
                 disagree on how a material is spelt.
"""

BACKBONES = ["mobilenet_v2", "resnet18", "densenet121"]
AUDITED = BACKBONES + ["cfd_pad"]

P_THRESHOLD = 0.30
NOISE_FLOOR = 1.2

SHORT = {"Consensual": "Cons.", "ScreenSpoof": "S.S.",
         "Body Double": "B.D.", "Elmer's Glue": "E.G."}


def short(name: str) -> str:
    """Table form of a PAI or material name."""
    for full, abbr in SHORT.items():
        name = name.replace(full, abbr)
    return name
