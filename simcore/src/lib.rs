pub mod vec;

//
//struct ForceTorque {
//    force: Vec3,
//    torque: Vec3,
//}
//
//trait FTModel {
//    fn init(&self);
//    fn step(&self) -> ForceTorque;
//}

pub fn add(left: u64, right: u64) -> u64 {
    left + right
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let result = add(2, 2);
        assert_eq!(result, 4);
    }
}
