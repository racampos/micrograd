use std::f64;
use std::fmt;
use std::ops::{Add, Div, Mul, Sub};

#[derive(Debug, Clone)]
enum Op {
    Add,
    Mul,
}

#[derive(Debug, Clone)]
struct Value {
    data: f64,
    _prev: Option<(Box<Value>, Box<Value>)>,
    _op: Option<Op>,
    grad: f64,
    _backward: Option<fn(Value) -> Value>,
}

impl Value {
    fn new(data: f64) -> Self {
        Self::new_ext(data, None, None, None)
    }

    fn new_ext(
        data: f64,
        _children: Option<(Box<Value>, Box<Value>)>,
        _op: Option<Op>,
        _backward: Option<fn(Value) -> Value>,
    ) -> Self {
        Self {
            data,
            _prev: _children,
            _op,
            grad: 0.0,
            _backward,
        }
    }

    fn tanh(self) -> Self {
        let x = self.data;
        let t = (f64::exp(2.0 * x) - 1.0) / (f64::exp(2.0 * x) + 1.0);
        Self::new_ext(
            t,
            Some((Box::new(self.clone()), Box::new(self.clone()))),
            None,
            None,
        )
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.data.fract() == 0.0 {
            write!(f, "Value(data={:.1})", self.data)
        } else {
            write!(f, "Value(data={})", self.data)
        }
    }
}

impl Add<Value> for Value {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self::new_ext(
            self.data + other.data,
            Some((Box::new(self), Box::new(other))),
            Some(Op::Add),
            Some(),
        )
    }
}

impl Sub<Value> for Value {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        Self::new_ext(
            self.data - other.data,
            Some((Box::new(self), Box::new(other))),
            Some(Op::Add),
        )
    }
}

impl Mul<Value> for Value {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        Self::new_ext(
            self.data * other.data,
            Some((Box::new(self), Box::new(other))),
            Some(Op::Mul),
        )
    }
}

impl Div<Value> for Value {
    type Output = Self;

    fn div(self, other: Self) -> Self {
        Self::new_ext(
            self.data / other.data,
            Some((Box::new(self), Box::new(other))),
            Some(Op::Mul),
        )
    }
}

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

    let mut x1w1 = x1.clone() * w1.clone();
    let mut x2w2 = x2.clone() * w2.clone();

    let mut x1w1x2w2 = x1w1.clone() + x2w2.clone();
    let mut n = x1w1x2w2.clone() + b.clone();

    let mut o = n.clone().tanh();

    x1.grad = w1.data * x1w1.grad;
    w1.grad = x1.data * x1w1.grad;
    x2.grad = w2.data * x2w2.grad;
    w2.grad = x2.data * x2w2.grad;

    x1w1.grad = 0.5;
    x2w2.grad = 0.5;

    x1w1x2w2.grad = 0.5;
    b.grad = 0.5;
    n.grad = 0.5;
    o.grad = 1.0;

    println!("{}", 1.0 - o.data.powf(2.0));
}
