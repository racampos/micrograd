use std::f64;
use std::fmt;
use std::ops::{Add, Div, Mul, Sub};

#[derive(Debug, Clone)]
enum Op {
    Add,
    Mul,
    Tanh,
}

#[derive(Debug, Clone)]
struct Value {
    data: f64,
    _prev: Option<(Box<Value>, Box<Value>)>,
    _op: Option<Op>,
    grad: f64,
}

const DUMMY_VALUE: Value = Value {
    data: 0.0,
    _prev: None,
    _op: None,
    grad: 0.0,
};

impl Value {
    fn new(data: f64) -> Self {
        Self::new_ext(data, None, None)
    }

    fn new_ext(data: f64, _children: Option<(Box<Value>, Box<Value>)>, _op: Option<Op>) -> Self {
        Self {
            data,
            _prev: _children,
            _op,
            grad: 0.0,
        }
    }

    fn tanh(self) -> Self {
        let x = self.data;
        let t = (f64::exp(2.0 * x) - 1.0) / (f64::exp(2.0 * x) + 1.0);
        Self::new_ext(
            t,
            Some((Box::new(self), Box::new(DUMMY_VALUE))),
            Some(Op::Tanh),
        )
    }

    fn _backward(&mut self) {
        if let Some(ref mut _prev) = self._prev {
            let (ref mut a, ref mut b) = *_prev;
            println!("{:?}", self._op);
            match self._op {
                Some(Op::Add) => {
                    a.grad += self.grad;
                    b.grad += self.grad;
                    println!("a.grad: {}, b.grad: {}", a.grad, b.grad);
                }
                Some(Op::Mul) => {
                    a.grad += b.data * self.grad;
                    b.grad += a.data * self.grad;
                }
                Some(Op::Tanh) => {
                    let t = (f64::exp(2.0 * a.data) - 1.0) / (f64::exp(2.0 * a.data) + 1.0);
                    a.grad = (1.0 - t.powf(2.0)) * self.grad;
                    println!("a.grad: {}", a.grad);
                }
                None => {}
            }
        }
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
fn main() {
    let x1 = Value::new(2.0);
    let x2 = Value::new(0.0);

    let w1 = Value::new(-3.0);
    let w2 = Value::new(1.0);

    let b = Value::new(6.8813735870195432);

    let x1w1 = x1 * w1;
    let x2w2 = x2 * w2;
    println!("x1w1: {}, x2w2: {}", x1w1, x2w2);

    let x1w1x2w2 = x1w1 + x2w2;
    println!("x1w1x2w2: {}", x1w1x2w2);

    let mut n = x1w1x2w2 + b;
    println!("n: {}", n);

    let mut o = n.tanh();
    println!("o: {}", o);

    o.grad = 1.0;
    o._backward();
    n._backward();
}
