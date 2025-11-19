use simcore::model::{ForceModel, Force};

pub struct Aero {
    pub aero_file: String
}

pub struct AeroStep {
    pub mach: f64,
    pub alpha_tot: f64,
    pub phi: f64
}

impl Aero {
    pub fn new(aero_table_file: String) -> Aero {
        Aero { aero_file: aero_table_file }
    }
}

impl ForceModel<AeroStep> for Aero {

    fn init(&self) {
        println!("Loading aero data from {}.", self.aero_file)
    }

    fn step(&self, step: &AeroStep) -> Force {
        let force_axial = 1.0 * step.alpha_tot;
        let aero_force = Force::new(force_axial, 0.0, 0.0);
        aero_force
    }

}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn can_instantiate_aero() {
        let aero = Aero::new("test.dat".to_string());
        assert_eq!(aero.aero_file, "test.dat".to_string());
    }

    #[test]
    fn can_take_step() {
        let aero = Aero::new("test.dat".to_string());
        let step_input = AeroStep { mach: 2.0, alpha_tot: 1.0, phi: 0.0 };
        let aero_force = aero.step(&step_input);
        assert!(aero_force.force.x.is_finite());
        assert!(aero_force.force.y.is_finite());
        assert!(aero_force.force.z.is_finite());
    }

}
