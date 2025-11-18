use kinematics;
use simcore::vec::Vec3;
use simcore::model::ForceModel;
use std::f64::consts::PI;
use aero;

fn main() {

    let v_a = Vec3::new(1.0, 0.0, 0.0);
    let axis = Vec3::new(0.0, 0.0, 1.0).normalize();
    let angle = 90.0 * PI / 180.0;
    let q_ba = kinematics::Quat::from_axis_angle(axis, angle);
    let v_b = q_ba.transform(v_a);

    println!("v_a = {v_a:?}");
    println!("q_ba = {q_ba:?}");
    println!("v_b = {v_b:?}");

    let aero_table = "aero.dat";
    let aero = aero::Aero::new(aero_table.to_string());
    aero.init();
    let aero_step = aero::AeroStep{
        mach: 2.5,
        alpha_tot: 90.0,
        phi: 0.0
    };

    let aero_force = aero.step(&aero_step);
    println!("aero_force = {aero_force:?}");


}
