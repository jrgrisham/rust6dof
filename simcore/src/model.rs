use crate::vec::Vec3;

/// Represents a force
#[derive(Debug, Copy, Clone)]
pub struct Force {
    pub force: Vec3
}

impl Force {

    /// Creates a new force
    ///
    /// # Example
    ///
    /// ```
    /// use simcore::model::Force;
    /// let f = Force::new(1.0, 2.0, 3.0);
    /// ```
    pub fn new(fx: f64, fy: f64, fz: f64) -> Force {
        Force { force: Vec3::new(fx, fy, fz) }
    }

}

pub trait ForceModel<T> {
    fn init(&self);
    fn step(&self, step_struct: &T) -> Force;
}
