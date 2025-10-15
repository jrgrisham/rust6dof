pub mod vec {

    /// Represents a vector of length 3.
    #[derive(Debug, Copy, Clone)]
    pub struct Vec3 {
        pub x: f64,
        pub y: f64,
        pub z: f64,
    }

    impl Vec3 {

        /// Creates a new vector
        ///
        /// # Example
        ///
        /// ```
        /// use simcore::vec::Vec3;
        /// let v = Vec3::new(1.0, 0.2, 0.3);
        /// ```
        pub fn new(xv: f64, yv: f64, zv: f64) -> Vec3 {
            Vec3 {
                x: xv,
                y: yv,
                z: zv,
            }
        }

        /// Dot product function
        ///
        /// # Example
        ///
        /// ```
        /// use simcore::vec::Vec3;
        /// let a = Vec3::new(1.0, 0.0, 0.0);
        /// let b = Vec3::new(0.5, 0.5, 0.25);
        /// let a_dot_b = a.dot(b);
        ///
        /// ```
        pub fn dot(self, other: Vec3) -> f64 {
            self.x * other.x + self.y * other.y + self.z * other.z
        }

        /// Computes cross product.
        pub fn cross(self, other: Vec3) -> Vec3 {
            Vec3 {
                x: self.y * other.z - self.z * other.y,
                y: self.z * other.x - self.x * other.z,
                z: self.x * other.y - self.y * other.x,
            }
        }

        /// Scales the present vector by the given factor.
        pub fn scale(self, factor: f64) -> Vec3 {
            Vec3 {
                x: self.x * factor,
                y: self.y * factor,
                z: self.z * factor,
            }
        }

        /// Element-wise addition of one vector with another.
        pub fn add(self, other: Vec3) -> Vec3 {
            Vec3 {
                x: self.x + other.x,
                y: self.y + other.y,
                z: self.z + other.z,
            }
        }

        /// Computes magnitude of vector.
        pub fn mag(&self) -> f64 {
            (self.x*self.x + self.y*self.y + self.z*self.z).sqrt()
        }

        /// Normalizes a vector.
        pub fn normalize(&self) -> Vec3 {
            let norm = self.mag();
            Vec3 {
                x: self.x / norm,
                y: self.y / norm,
                z: self.z / norm,
            }
        }
    }

}

#[cfg(test)]
mod tests {

    use super::*;
    use vec::Vec3;

    #[test]
    fn can_instantiate_vec() {
        let v = Vec3::new(1.0, 2.0, 3.0);
        assert_eq!(v.x, 1.0);
        assert_eq!(v.y, 2.0);
        assert_eq!(v.z, 3.0);
    }

    #[test]
    fn can_dot_vec1() {
        let v1 = Vec3::new(1.0, 0.0, 0.0);
        let v2 = Vec3::new(0.0, 1.0, 0.0);
        let v3 = v1.dot(v2);
        assert_eq!(v3,0.0);
    }

    #[test]
    fn can_dot_vec2() {
        let v1 = Vec3::new(1.0, 0.3, 0.0);
        let v2 = Vec3::new(0.5, 1.0, 0.0);
        let v3 = v1.dot(v2);
        assert_eq!(v3,0.8);
    }

    #[test]
    fn can_cross_vec1() {
        let v1 = Vec3::new(1.0, 0.0, 0.0);
        let v2 = Vec3::new(0.0, 1.0, 0.0);
        let v3 = v1.cross(v2);
        assert_eq!(v3.x,0.0);
        assert_eq!(v3.y,0.0);
        assert_eq!(v3.z,1.0);
    }

    #[test]
    fn can_cross_vec2() {
        let v1 = Vec3::new(1.0, 0.0, 0.0);
        let v2 = Vec3::new(1.0, 0.0, 0.0);
        let v3 = v1.cross(v2);
        assert_eq!(v3.x,0.0);
        assert_eq!(v3.y,0.0);
        assert_eq!(v3.z,0.0);
    }

    #[test]
    fn can_compute_vec_mag() {
        let q0 = 0.0;
        let q1 = 1.0;
        let q2 = 2.0;
        let v1 = Vec3::new(q0, q1, q2);
        let v2 = vec![q0, q1, q2];
        let v2mag = v2
            .iter()
            .map(|x| x * x)
            .reduce(|acc, x| acc + x)
            .expect("Unable to compute sum of squares.")
            .sqrt();
        assert_eq!(v1.mag(), v2mag);
    }

    #[test]
    fn can_normalize_vec() {
        let s0 = 0.0;
        let s1 = 1.0;
        let s2 = 2.0;
        let v1 = Vec3::new(s0, s1, s2);
        let v2 = vec![s0, s1, s2];
        let v2mag = v2
            .iter()
            .map(|x| x * x)
            .reduce(|acc, x| acc + x)
            .expect("Unable to compute sum of squares.")
            .sqrt();

        let v1norm = v1.normalize();
        let v2norm = vec![s0 / v2mag, s1 / v2mag, s2 / v2mag];

        assert_eq!(v1norm.x, v2norm[0]);
        assert_eq!(v1norm.y, v2norm[1]);
        assert_eq!(v1norm.z, v2norm[2]);
    }

}
