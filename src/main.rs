use std::cell::RefCell;
use std::f64;
use std::fmt;
use std::rc::Rc;

#[derive(Debug, Clone, PartialEq)]
enum Op {
    Add,
    Mul,
    Tanh,
}

#[derive(Debug, Clone, PartialEq)]
struct _Value {
    data: f64,
    _prev: Option<(Value, Value)>,
    _op: Option<Op>,
    grad: f64,
}

#[derive(Debug, Clone, PartialEq)]
struct Value(Rc<RefCell<_Value>>);

impl Value {
    fn new(data: f64) -> Self {
        Self::new_ext(data, None, None)
    }

    fn new_ext(data: f64, _children: Option<(Value, Value)>, _op: Option<Op>) -> Self {
        Value(Rc::new(RefCell::new(_Value {
            data,
            _prev: _children,
            _op,
            grad: 0.0,
        })))
    }

    pub fn clone_inner(&self) -> Rc<RefCell<_Value>> {
        Rc::clone(&self.0)
    }

    pub fn update_data(&self, new_data: f64) {
        let mut inner = self.0.borrow_mut();
        inner.data = new_data;
    }

    pub fn update_grad(&self, new_grad: f64) {
        let mut inner = self.0.borrow_mut();
        inner.grad = new_grad;
    }

    pub fn get_data(&self) -> f64 {
        self.0.borrow().data
    }

    pub fn get_prev(&self) -> Option<(Value, Value)> {
        self.0.borrow()._prev.clone()
    }

    pub fn get_op(&self) -> Option<Op> {
        self.0.borrow()._op.clone()
    }

    pub fn get_grad(&self) -> f64 {
        self.0.borrow().grad
    }

    fn tanh(self) -> Self {
        let t = self.get_data().tanh();
        Self::new_ext(t, Some((self.clone(), self.clone())), Some(Op::Tanh))
    }

    fn add(self, other: Self) -> Self {
        Self::new_ext(
            self.get_data() + other.get_data(),
            Some((self.clone(), other.clone())),
            Some(Op::Add),
        )
    }

    fn mul(self, other: Self) -> Self {
        Self::new_ext(
            self.get_data() * other.get_data(),
            Some((self.clone(), other.clone())),
            Some(Op::Mul),
        )
    }

    fn sub(self, other: Self) -> Self {
        Self::new_ext(
            self.get_data() - other.get_data(),
            Some((self.clone(), other.clone())),
            Some(Op::Add),
        )
    }

    fn div(self, other: Self) -> Self {
        Self::new_ext(
            self.get_data() / other.get_data(),
            Some((self.clone(), other.clone())),
            Some(Op::Mul),
        )
    }

    fn _backward(self) {
        if let Some(ref _prev) = self.get_prev() {
            let (ref a, ref b) = *_prev;
            // println!("{:?}", self.get_op());
            match self.get_op() {
                Some(Op::Add) => {
                    a.update_grad(a.get_grad() + self.get_grad());
                    b.update_grad(b.get_grad() + self.get_grad());
                    // println!("a.grad: {}, b.grad: {}", a.get_grad(), b.get_grad());
                }
                Some(Op::Mul) => {
                    a.update_grad(a.get_grad() + b.get_data() * self.get_grad());
                    b.update_grad(b.get_grad() + a.get_data() * self.get_grad());
                    // println!("a.grad: {}, b.grad: {}", a.get_grad(), b.get_grad());
                }
                Some(Op::Tanh) => {
                    let t = a.get_data().tanh();
                    a.update_grad((1.0 - t.powf(2.0)) * self.get_grad());
                    // println!("a.grad: {}", a.get_grad());
                }
                None => {}
            }
        }
    }

    fn backward(self) {
        let mut topo: Vec<Value> = vec![];
        let mut visited: Vec<Value> = vec![];
        fn build_topo(v: &Value, topo: &mut Vec<Value>, visited: &mut Vec<Value>) {
            if !visited.iter().any(|x| x == v) {
                visited.push(v.clone());
                if let Some(ref _prev) = v.get_prev() {
                    build_topo(&_prev.0, topo, visited);
                    build_topo(&_prev.1, topo, visited);
                }
                topo.push(v.clone());
            }
        }
        build_topo(&self, &mut topo, &mut visited);

        self.update_grad(1.0);
        topo.reverse();
        for node in topo {
            node._backward();
        }
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.get_data().fract() == 0.0 {
            write!(f, "Value(data={:.1})", self.get_data())
        } else {
            write!(f, "Value(data={})", self.get_data())
        }
    }
}

fn main() {
    let x1 = Value::new(2.0);
    let x2 = Value::new(0.0);

    let w1 = Value::new(-3.0);
    let w2 = Value::new(1.0);

    let b = Value::new(6.8813735870195432);

    let x1w1 = x1.clone().mul(w1.clone());
    let x2w2 = x2.clone().mul(w2.clone());

    let x1w1x2w2 = x1w1.clone().add(x2w2.clone());

    let n = x1w1x2w2.clone().add(b.clone());

    let o = n.clone().tanh();
    o.backward();

    // o.update_grad(1.0);
    // o._backward();
    println!("n.grad: {}", n.get_grad());
    // n._backward();
    println!("x1w1x2w2.grad: {}", x1w1x2w2.get_grad());
    println!("b.grad: {}", b.get_grad());
    // b._backward();
    // x1w1x2w2._backward();
    println!("w1x1.grad: {}", x1w1.get_grad());
    println!("w2x2.grad: {}", x2w2.get_grad());
    // x2w2._backward();
    // x1w1._backward();
    println!("w1.grad: {}", w1.get_grad());
    println!("w2.grad: {}", w2.get_grad());
    println!("x1.grad: {}", x1.get_grad());
    println!("x2.grad: {}", x2.get_grad());
}
