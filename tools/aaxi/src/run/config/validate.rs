use crate::run::config::{Config, Interval, Strategy};

impl Config {
    /// Check that configuration file satisfies requirements.
    pub fn validate(&self) -> eyre::Result<()> {
        // ************************
        // Domain

        eyre::ensure!(
            self.domain.radius > 0.0 && self.domain.height > 0.0,
            "domain must have positive non-zero radius and height"
        );

        eyre::ensure!(
            self.domain.cell_size >= 2 * self.domain.cell_ghost,
            "domain cell nodes must be >= 2 * ghost"
        );

        eyre::ensure!(
            self.domain.global_refine < self.limits.max_levels,
            "global_refine must be < max_levels"
        );

        // *************************
        // Initial

        eyre::ensure!(self.initial.relax.cfl > 0.0, "initial cfl must be positive");
        eyre::ensure!(
            self.initial.relax.dampening > 0.0,
            "initial dampening must be positive"
        );
        eyre::ensure!(
            self.initial.relax.tolerance > 0.0,
            "initial error must be positive"
        );

        // *************************
        // Evolve

        eyre::ensure!(self.evolve.cfl > 0.0, "evolve cfl must be positive");

        // **************************
        // Visualize

        eyre::ensure!(
            matches!(
                self.visualize.initial_relax_interval,
                Interval::Steps { .. }
            ),
            "relaxation intervals must be measured in steps"
        );

        eyre::ensure!(
            matches!(
                self.visualize.horizon_relax_interval,
                Interval::Steps { .. }
            ),
            "relaxation intervals must be measured in steps"
        );

        // **************************
        // Cache

        // **************************
        // Error handler

        eyre::ensure!(
            matches!(
                self.error_handler.on_max_nodes,
                Strategy::Collapse | Strategy::Disperse | Strategy::Crash
            ),
            "max_nodes error can not be ignored"
        );

        eyre::ensure!(
            matches!(
                self.error_handler.on_max_memory,
                Strategy::Collapse | Strategy::Disperse | Strategy::Crash
            ),
            "max_memory error can not be ignored"
        );

        eyre::ensure!(
            matches!(
                self.error_handler.on_max_initial_steps,
                Strategy::Crash | Strategy::Ignore
            ),
            "max_initial_steps error error must trigger crash or be ignored"
        );

        eyre::ensure!(
            matches!(
                self.error_handler.on_max_evolve_steps,
                Strategy::Collapse | Strategy::Disperse | Strategy::Crash
            ),
            "max_evolve_steps error can not be ignored"
        );

        eyre::ensure!(
            matches!(
                self.error_handler.on_max_evolve_proper_time,
                Strategy::Collapse | Strategy::Disperse | Strategy::Crash
            ),
            "max_evolve_proper_time error can not be ignored"
        );

        eyre::ensure!(
            matches!(
                self.error_handler.on_max_evolve_coord_time,
                Strategy::Collapse | Strategy::Disperse | Strategy::Crash
            ),
            "max_evolve_coord_time error can not be ignored"
        );

        // ***********************************
        // Horizon

        Ok(())
    }
}
