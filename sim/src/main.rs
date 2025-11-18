use kinematics;
use simcore::vec::Vec3;
use std::f64::consts::PI;

fn main() {

    let v_a = Vec3::new(1.0, 0.0, 0.0);
    let axis = Vec3::new(0.0, 0.0, 1.0).normalize();
    let angle = 90.0 * PI / 180.0;
    let q_ba = kinematics::Quat::from_axis_angle(axis, angle);
    let v_b = q_ba.transform(v_a);

    println!("v_a = {v_a:?}");
    println!("q_ba = {q_ba:?}");
    println!("v_b = {v_b:?}");

}
