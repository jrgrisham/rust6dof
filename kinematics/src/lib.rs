use simcore::vec::Vec3;

/// Represents a unit quaternion.
#[derive(Debug, Copy, Clone)]
pub struct Quat {
    w: f64,
    x: f64,
    y: f64,
    z: f64,
}

impl Quat {

    pub fn new(wv: f64, xv: f64, yv: f64, zv: f64) -> Quat{
        Quat{
            w: wv,
            x: xv,
            y: yv,
            z: zv,
        }
    }

    /// Rotates a vector given a quaternion and vector
    pub fn rotate(&self, v: Vec3) -> Vec3 {
        let qv = Vec3 { x: self.x, y: self.y, z: self.z };
        let uv = qv.cross(v);
        let uuv = qv.cross(uv);
        v.add(uv.scale(2.0 * self.w)).add(uuv.scale(2.0))
    }

    /// Transforms coordinates of the vector.
    pub fn transform(&self, v: Vec3) -> Vec3 {
        self.conjugate().rotate(v)
    }

    /// Performs quaternion multiplication
    pub fn multiply(&self, other: &Quat) -> Quat {
        Quat {
            w: self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z,
            x: self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y,
            y: self.w * other.y - self.x * other.z + self.y * other.w + self.z * other.x,
            z: self.w * other.z + self.x * other.y - self.y * other.x + self.z * other.w,
        }
    }

    /// Creates a quaternion from the given axis and angle
    pub fn from_axis_angle(axis: Vec3, angle_rad: f64) -> Quat {
        let half_angle = angle_rad * 0.5;
        let sin_half = half_angle.sin();
        Quat {
            w: half_angle.cos(),
            x: axis.x * sin_half,
            y: axis.y * sin_half,
            z: axis.z * sin_half,
        }
    }

    /// Conjugates the given quaternion (assuming a unit quaternion)
    pub fn conjugate(&self) -> Quat {
        Quat {
            w: self.w,
            x: -self.x,
            y: -self.y,
            z: -self.z,
        }
    }

    /// Computes quaternion magnitude
    pub fn mag(&self) -> f64 {
        (self.w*self.w + self.x*self.x + self.y*self.y + self.z*self.z).sqrt()
    }

    /// Normalizes quaternion
    pub fn normalize(&self) -> Quat {
        let norm = Quat::mag(self);
        Quat {
            w: self.w / norm,
            x: self.x / norm,
            y: self.y / norm,
            z: self.z / norm,
        }
    }

}

/// Struct to represent kinematic state for a 6DOF body.
#[derive(Debug, Copy, Clone)]
pub struct State {
    pos: Vec3,
    vel: Vec3,
    acc: Vec3,
    quat: Quat,
    omega: Vec3,
    alpha: Vec3,
}

impl State {

    /// Creates a new kinematic state.
    pub fn new(p: Vec3, v: Vec3, a: Vec3, q: Quat, w: Vec3, al: Vec3) -> State {
        Self {
            pos: p,
            vel: v,
            acc: a,
            quat: q,
            omega: w,
            alpha: al,
        }
    }

    /// Transforms kinematic state coordinates using the given quaternion.
    pub fn transform(&self, q: Quat) -> State {

        // Perform coordinate transformation.
        let pos = q.transform(self.pos);
        let vel = q.transform(self.vel);
        let acc = q.transform(self.acc);
        let quat = self.quat.clone();
        let omega = q.transform(self.omega);
        let alpha = q.transform(self.alpha);

        Self {
            pos,
            vel,
            acc,
            quat,
            omega,
            alpha
        }

    }

}

#[cfg(test)]
mod tests {

    use super::*;
    use std::f64::consts::PI;
    use assert_approx_eq::assert_approx_eq;

    const TOL: f64 = 1e-15;

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


    #[test]
    fn can_instantiate_quat() {
        let q = Quat{
            w: 1.0,
            x: 0.0,
            y: 0.0,
            z: 0.0};
        assert_eq!(q.w, 1.0);
        assert_eq!(q.x, 0.0);
        assert_eq!(q.y, 0.0);
        assert_eq!(q.z, 0.0);
    }

    #[test]
    fn can_conjugate() {
        let q = Quat{
            w: 1.0,
            x: 2.0,
            y: 3.0,
            z: 4.0};
        let qconj = q.conjugate();
        assert_eq!(qconj.w, 1.0);
        assert_eq!(qconj.x, -2.0);
        assert_eq!(qconj.y, -3.0);
        assert_eq!(qconj.z, -4.0);
    }

    #[test]
    fn can_multiply() {
        let axis = Vec3::new(1.0, 0.0, 0.0);
        let angle = 90.0 * PI / 180.0;
        let q1 = Quat::from_axis_angle(axis, angle);
        let q1conj = q1.conjugate();
        let q2 = q1.multiply(&q1conj);
        assert_eq!(q2.w, 1.0);
        assert_eq!(q2.x, 0.0);
        assert_eq!(q2.y, 0.0);
        assert_eq!(q2.z, 0.0);
    }

    #[test]
    fn can_create_from_axis_angle() {
        let axis = Vec3::new(1.0, 0.0, 0.0);
        let angle = 90.0 * PI / 180.0;
        let q = Quat::from_axis_angle(axis, angle);
        let halfangle = angle * 0.5;
        assert_eq!(q.w, halfangle.cos());
        assert_eq!(q.x, axis.x*halfangle.sin());
        assert_eq!(q.y, axis.y*halfangle.sin());
        assert_eq!(q.z, axis.z*halfangle.sin());
    }

    #[test]
    fn can_compute_quat_mag() {
        let q0 = 1.0;
        let q1 = 2.0;
        let q2 = 3.0;
        let q3 = 4.0;
        let q = Quat::new(q0, q1, q2, q3);
        let v = vec![q0, q1, q2, q3];
        let expected = v
            .iter()
            .map(|x| x*x)
            .reduce(|acc, x| acc + x)
            .expect("Unable to compute sum of squares")
            .sqrt();
        assert_eq!(q.mag(), expected);
    }

    #[test]
    fn can_normalize_quat() {
        let q0 = 1.0;
        let q1 = 2.0;
        let q2 = 3.0;
        let q3 = 4.0;
        let q = Quat::new(q0, q1, q2, q3);
        let v = vec![q0, q1, q2, q3];
        let mag = v
            .iter()
            .map(|x| x*x)
            .reduce(|acc, x| acc + x)
            .expect("Unable to compute sum of squares")
            .sqrt();
        let qnorm1 = q.normalize();
        let qnorm2 = vec![q0 / mag, q1 / mag, q2 / mag, q3 / mag];
        assert_eq!(qnorm1.w,qnorm2[0]);
        assert_eq!(qnorm1.x,qnorm2[1]);
        assert_eq!(qnorm1.y,qnorm2[2]);
        assert_eq!(qnorm1.z,qnorm2[3]);
    }

    #[test]
    fn can_rotate_vec() {
        let axis = Vec3::new(0.0, 0.0, 1.0);
        let angle = 90.0*PI/180.0;
        let q = Quat::from_axis_angle(axis,angle);
        let v = Vec3::new(1.0, 0.0, 0.0);
        let vrot = q.rotate(v);
        assert!(vrot.x.abs()<1.0e-14);
        assert_eq!(vrot.y,1.0);
        assert!(vrot.z.abs()<1.0e-14);
    }

    #[test]
    fn can_transform_vec() {
        let axis = Vec3::new(0.0, 0.0, 1.0);
        let angle = 90.0*PI/180.0;
        let q = Quat::from_axis_angle(axis,angle);
        let v = Vec3::new(1.0, 0.0, 0.0);
        let vrot = q.transform(v);
        assert!(vrot.x.abs()<1.0e-14);
        assert_eq!(vrot.y,-1.0);
        assert!(vrot.z.abs()<1.0e-14);
    }

    #[test]
    fn can_transform_state() {

        // Build quaternion.
        let axis = Vec3::new(0.0, 0.0, 1.0);
        let angle = 90.0*PI/180.0;
        let q_21 = Quat::from_axis_angle(axis,angle);

        // Build state.
        let v = Vec3::new(1.0, 0.0, 0.0);
        let x_21_1 = State::new(v, v, v, q_21, v, v);

        // Perform coordinate transformation.
        let x_21_2 = x_21_1.transform(q_21);

        // Check coordinate transformation.
        assert_approx_eq!(x_21_2.pos.x   , 0.0          , TOL);
        assert_approx_eq!(x_21_2.pos.y   , -1.0         , TOL);
        assert_approx_eq!(x_21_2.pos.z   , 0.0          , TOL);
        assert_approx_eq!(x_21_2.vel.x   , x_21_2.pos.x , TOL);
        assert_approx_eq!(x_21_2.vel.y   , x_21_2.pos.y , TOL);
        assert_approx_eq!(x_21_2.vel.z   , x_21_2.pos.z , TOL);
        assert_approx_eq!(x_21_2.acc.x   , x_21_2.pos.x , TOL);
        assert_approx_eq!(x_21_2.acc.y   , x_21_2.pos.y , TOL);
        assert_approx_eq!(x_21_2.acc.z   , x_21_2.pos.z , TOL);
        assert_approx_eq!(x_21_2.omega.x , x_21_2.pos.x , TOL);
        assert_approx_eq!(x_21_2.omega.y , x_21_2.pos.y , TOL);
        assert_approx_eq!(x_21_2.omega.z , x_21_2.pos.z , TOL);
        assert_approx_eq!(x_21_2.alpha.x , x_21_2.pos.x , TOL);
        assert_approx_eq!(x_21_2.alpha.y , x_21_2.pos.y , TOL);
        assert_approx_eq!(x_21_2.alpha.z , x_21_2.pos.z , TOL);
        assert_approx_eq!(x_21_2.quat.w  , q_21.w       , TOL);
        assert_approx_eq!(x_21_2.quat.x  , q_21.x       , TOL);
        assert_approx_eq!(x_21_2.quat.y  , q_21.y       , TOL);
        assert_approx_eq!(x_21_2.quat.z  , q_21.z       , TOL);

    }

}
