use std::f64;
use std::fmt;

#[derive(Debug)]
enum Op {
    Add,
    Mul,
    Tanh,
}

#[derive(Debug)]
struct Value<'a> {
    data: f64,
    _prev: Option<(Box<&'a mut Value<'a>>, Box<&'a mut Value<'a>>)>,
    _op: Option<Op>,
    grad: f64,
}

const DUMMY_VALUE: Value = Value {
    data: 0.0,
    _prev: None,
    _op: None,
    grad: 0.0,
};

impl<'a> Value<'a> {
    fn new(data: f64) -> Self {
        Self::new_ext(data, None, None)
    }

    fn new_ext(
        data: f64,
        _children: Option<(Box<&'a mut Value<'a>>, Box<&'a mut Value<'a>>)>,
        _op: Option<Op>,
    ) -> Self {
        Self {
            data,
            _prev: _children,
            _op,
            grad: 0.0,
        }
    }

    fn _backward(&mut self) {
        if let Some(ref mut _prev) = self._prev {
            let (ref mut a, ref mut b) = *_prev;
            match self._op {
                Some(Op::Add) => {
                    a.grad += self.grad;
                    b.grad += self.grad;
                }
                Some(Op::Mul) => {
                    a.grad += b.data * self.grad;
                    b.grad += a.data * self.grad;
                }
                Some(Op::Tanh) => {
                    let t = (f64::exp(2.0 * self.data) - 1.0) / (f64::exp(2.0 * self.data) + 1.0);
                    a.grad += (1.0 - t.powf(2.0)) * self.grad;
                }
                None => {}
            }
        }
    }

    fn set_grad(&mut self, grad: f64) {
        self.grad = grad;
    }

    fn add(&mut self, other: &mut Value<'a>) -> Self {
        Self::new_ext(
            self.data + other.data,
            Some((Box::new(self), Box::new(other))),
            Some(Op::Add),
        )
    }

    fn mul(&mut self, other: &mut Value<'a>) -> Self {
        Self::new_ext(
            self.data * other.data,
            Some((Box::new(self), Box::new(other))),
            Some(Op::Mul),
        )
    }

    fn tanh(&mut self) -> Self {
        let x = self.data;
        let t = (f64::exp(2.0 * x) - 1.0) / (f64::exp(2.0 * x) + 1.0);
        Self::new_ext(t, Some((Box::new(self), Box::new(&mut DUMMY_VALUE))), None)
    }
}

impl<'a> fmt::Display for Value<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.data.fract() == 0.0 {
            write!(f, "Value(data={:.1})", self.data)
        } else {
            write!(f, "Value(data={})", self.data)
        }
    }
}

// impl<'a> Add<Value<'a>> for &'a Value<'a> {
//     type Output = Value<'a>;

//     fn add(self, other: Self) -> Self::Output {
//         Self::new_ext(
//             self.data + other.data,
//             Some((Box::new(self), Box::new(other))),
//             Some(Op::Add),
//         )
//     }
// }

// impl Sub<Value> for Value {
//     type Output = Self;

//     fn sub(self, other: Self) -> Self {
//         Self::new_ext(
//             self.data - other.data,
//             Some((Box::new(self), Box::new(other))),
//             Some(Op::Add),
//         )
//     }
// }

// impl Mul<Value> for Value {
//     type Output = Self;

//     fn mul(self, other: Self) -> Self {
//         Self::new_ext(
//             self.data * other.data,
//             Some((Box::new(self), Box::new(other))),
//             Some(Op::Mul),
//         )
//     }
// }

// impl Div<Value> for Value {
//     type Output = Self;

//     fn div(self, other: Self) -> Self {
//         Self::new_ext(
//             self.data / other.data,
//             Some((Box::new(self), Box::new(other))),
//             Some(Op::Mul),
//         )
//     }
// }

// let mut a = Value::new(2.0);
// let mut b = Value::new(-3.0);
// let mut c = Value::new(10.0);
// let mut e = a.clone() * b.clone();
// let mut d = e.clone() + c.clone();
// let mut f = Value::new(-2.0);
// let mut L = d.clone()*f.clone();

// L.grad = 1.0;
// f.grad = 4.0;
// d.grad = -2.0;
// e.grad = -2.0;
// c.grad = -2.0;
// a.grad = 6.0;
// b.grad = -4.0;

// a.data += 0.01 * a.grad;
// b.data += 0.01 * b.grad;
// c.data += 0.01 * c.grad;
// f.data += 0.01 * f.grad;

// e = a * b;
// d = e + c;
// L = d * f;

// println!("{}", L.data);

// *******

// let h = 0.0001;

// let a = Value::new(2.0);
// let b = Value::new(-3.0);
// let c = Value::new(10.0);
// let e = a*b;
// let d = e + c;
// let f = Value::new(-2.0);
// let L = d*f;
// let L1 = L.data;

// let a = Value::new(2.0 + h);
// let b = Value::new(-3.0);
// let c = Value::new(10.0);
// let e = a*b;
// let d = e + c;
// let f = Value::new(-2.0);
// let L = d*f;
// let L2 = L.data;

// println!("{}", ((L2 - L1)/h));
fn main() {
    let mut x1 = Value::new(2.0);
    let mut x2 = Value::new(0.0);

    let mut w1 = Value::new(-3.0);
    let mut w2 = Value::new(1.0);

    let mut b = Value::new(6.8813735870195432);

    let mut x1w1 = x1.mul(&mut w1);
    let mut x2w2 = x2.mul(&mut w2);

    let mut x1w1x2w2 = x1w1.add(&mut x2w2);
    let mut n = x1w1x2w2.add(&mut b);

    let mut o = n.tanh();

    // x1.grad = w1.data * x1w1.grad;
    // w1.grad = x1.data * x1w1.grad;
    // x2.grad = w2.data * x2w2.grad;
    // w2.grad = x2.data * x2w2.grad;

    // x1w1.grad = 0.5;
    // x2w2.grad = 0.5;

    // x1w1x2w2.grad = 0.5;
    // b.grad = 0.5;
    // n.grad = 0.5;
    // o.grad = 1.0;

    o.set_grad(1.0);
    o._backward();

    println!("{}", n.grad);
}
