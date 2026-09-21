"""Legacy figure API adapter. Prefer dap.DensityAwareForest for new work."""

from dap import DensityAwareForest


class BaseIForest(DensityAwareForest):
    family = "if"
    default_mode = "IF"

    def __init__(
        self,
        *,
        n_trees=100,
        sample_size=128,
        max_depth=None,
        mode="",
        alpha=1.0,
        random_seed=0,
        n_dir_candidates=256,
    ):
        selected = mode or self.default_mode
        allowed = {
            "if": {"IF", "IF+DAS"},
            "eif": {"EIF", "EIF+DAS", "EIF+DAD", "EIF+DAS+DAD"},
            "scif": {"SCIF", "SCIF+DAD"},
        }
        if selected not in allowed[self.family]:
            raise ValueError(f"Invalid mode: {selected}")
        self.n_trees, self.sample_size, self.mode = n_trees, sample_size, mode
        self.random_seed, self.n_dir_candidates = random_seed, n_dir_candidates
        super().__init__(
            model=self.family,
            das="DAS" in selected,
            dad="DAD" in selected,
            n_estimators=n_trees,
            max_samples=sample_size,
            max_depth=max_depth,
            alpha=alpha,
            random_state=random_seed,
            n_directions=n_dir_candidates,
        )
