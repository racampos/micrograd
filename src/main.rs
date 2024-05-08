use rand::distributions::Uniform;
use rand::Rng;
use std::cell::RefCell;
use std::collections::btree_map::Range;
use std::f64;
use std::fmt;
use std::rc::Rc;
use std::vec;

#[derive(Debug, Clone, PartialEq)]
enum Op {
    Add,
    Mul,
    Tanh,
    Exp,
    Pow,
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
        Self::new_ext(
            self.get_data().tanh(),
            Some((self.clone(), self.clone())),
            Some(Op::Tanh),
        )
    }

    fn exp(self) -> Self {
        Self::new_ext(
            self.get_data().exp(),
            Some((self.clone(), self.clone())),
            Some(Op::Exp),
        )
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

    fn neg(self) -> Self {
        self.mul(Value::new(-1.0))
    }

    fn sub(self, other: Self) -> Self {
        self.add(other.neg())
    }

    fn pow(self, other: Self) -> Self {
        Self::new_ext(
            self.get_data().powf(other.get_data()),
            Some((self.clone(), other.clone())),
            Some(Op::Pow),
        )
    }

    fn div(self, other: Self) -> Self {
        self.mul(other.pow(Value::new(-1.0)))
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
                Some(Op::Exp) => {
                    a.update_grad(a.get_data().exp() * self.get_grad());
                    // println!("a.grad: {}", a.get_grad());
                }
                Some(Op::Pow) => {
                    a.update_grad(
                        b.get_data() * a.get_data().powf(b.get_data() - 1.0) * self.get_grad(),
                    );
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

struct Neuron {
    w: Vec<Value>,
    b: Value,
}

impl Neuron {
    fn new(nin: u16) -> Self {
        let mut rng = rand::thread_rng();
        let range = Uniform::new(-1.0, 1.0);

        let w: Vec<Value> = (0..nin).map(|_| Value::new(rng.sample(&range))).collect();

        let b = Value::new(rng.sample(&range));

        Neuron { w, b }
    }

    pub fn call(&self, inputs: Vec<Value>) -> Value {
        assert_eq!(
            self.w.len(),
            inputs.len(),
            "Input size must match number of weights."
        );
        // let inputs: Vec<Value> = inputs.iter()
        //     .map(|&x| Value::new(x))
        //     .collect();
        let wx = self
            .w
            .iter()
            .zip(inputs.iter())
            .map(|(weight, &ref input)| weight.clone().mul(input.clone()));

        let act = wx
            .into_iter()
            .fold(Value::new(0.0), |acc, x| acc.add(x))
            .add(self.b.clone());
        act.tanh()
    }

    fn parameters(&self) -> Vec<Value> {
        let mut params: Vec<Value> = self.w.clone();
        params.push(self.b.clone());
        params
    }
}

struct Layer {
    neurons: Vec<Neuron>,
}

impl Layer {
    fn new(nin: u16, nout: u16) -> Self {
        let neurons: Vec<Neuron> = (0..nout).map(|_| Neuron::new(nin)).collect();
        Layer { neurons }
    }

    pub fn call(&self, inputs: Vec<Value>) -> Vec<Value> {
        self.neurons
            .iter()
            .map(|neuron| neuron.call(inputs.clone()))
            .collect()
    }

    fn parameters(&self) -> Vec<Value> {
        self.neurons
            .iter()
            .map(|neuron| neuron.parameters())
            .flatten()
            .collect()
    }
}

struct MLP {
    layers: Vec<Layer>,
}

impl MLP {
    fn new(nin: u16, nouts: Vec<u16>) -> Self {
        let sz = vec![vec![nin], nouts].concat();
        let layers: Vec<Layer> = sz.windows(2).map(|w| Layer::new(w[0], w[1])).collect();
        MLP { layers }
    }

    pub fn call(&self, inputs: &[f64]) -> Value {
        let inputs: Vec<Value> = inputs.iter().map(|&x| Value::new(x)).collect();
        let out = self
            .layers
            .iter()
            .fold(inputs.to_vec(), |acc, layer| layer.call(acc));
        out[0].clone()
    }

    fn parameters(&self) -> Vec<Value> {
        self.layers
            .iter()
            .map(|layer| layer.parameters())
            .flatten()
            .collect()
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
    // println!("n.grad: {}", n.get_grad());
    // n._backward();
    // println!("x1w1x2w2.grad: {}", x1w1x2w2.get_grad());
    // println!("b.grad: {}", b.get_grad());
    // b._backward();
    // x1w1x2w2._backward();
    // println!("w1x1.grad: {}", x1w1.get_grad());
    // println!("w2x2.grad: {}", x2w2.get_grad());
    // x2w2._backward();
    // x1w1._backward();
    // println!("w1.grad: {}", w1.get_grad());
    // println!("w2.grad: {}", w2.get_grad());
    // println!("x1.grad: {}", x1.get_grad());
    // println!("x2.grad: {}", x2.get_grad());

    // MLP Training

    let x = [2.0, 3.0, -1.0];
    let n = MLP::new(3, [4, 4, 1].to_vec());

    let xs = [
        [2.0, 3.0, -1.0],
        [3.0, -1.0, 5.0],
        [0.5, 1.0, 1.0],
        [1.0, 1.0, -1.0],
    ];
    let ys = [1.0, -1.0, -1.0, 1.0];

    let ypred: Vec<Value> = xs.iter().map(|row| n.call(row)).collect();
    println!("\nypred before training:\n");
    for y in ypred {
        println!("{}", y.get_data());
    }
    println!("\nTraining...");
    for _k in 0..100 {
        // Forward pass
        let ypred: Vec<Value> = xs.iter().map(|row| n.call(row)).collect();
        let squared_differences: Vec<Value> = ys
            .iter()
            .zip(ypred.iter())
            .map(|(&ygt, &ref yout)| (yout.clone().sub(Value::new(ygt))).pow(Value::new(2.0)))
            .collect();
        let loss = squared_differences
            .iter()
            .fold(Value::new(0.0), |acc, x| acc.add(x.clone()));

        // Print loss
        println!("loss: {}", loss.get_data());

        // Backward pass
        for p in n.parameters() {
            p.update_grad(0.0);
        }
        loss.clone().backward();

        // Update parameters
        for p in n.parameters() {
            p.update_data(p.get_data() - 0.1 * p.get_grad());
        }
    }
    let ypred: Vec<Value> = xs.iter().map(|row| n.call(row)).collect();
    println!("\nypred after training:\n");
    for y in ypred {
        println!("{}", y.get_data());
    }
}
