use crate::tensor::{float_eq, Tensor};

// get (row) vectors from a 2D table given a list of indices
pub fn gather(y: &mut Tensor<f32>, indices: &Tensor<u32>, table: &Tensor<f32>) {
    let length = indices.size();
    let table_shape = table.shape();
    assert!(table_shape.len() == 2);
    let dim = table_shape[1];
    assert!(y.size() == length * dim);
    for i in 0..length {
        let src = &table.data()[indices.data()[i] as usize * dim..][..dim];
        let dst = &mut unsafe { y.data_mut() }[i * dim..][..dim];
        dst.copy_from_slice(src);
    }
}

// RoPE: Rotary Positional Embedding
pub fn rope(y: &mut Tensor<f32>, start_pos: usize, theta: f32) {
    let shape = y.shape();
    assert!(shape.len() == 3);
    let seq_len = shape[0];
    let n_heads = shape[1];
    let d = shape[2];
    let data = unsafe { y.data_mut() };
    for tok in 0..seq_len {
        let pos = start_pos + tok;
        for head in 0..n_heads {
            for i in 0..d / 2 {
                let a = data[tok * n_heads * d + head * d + i];
                let b = data[tok * n_heads * d + head * d + i + d / 2];
                let freq = pos as f32 / theta.powf((i * 2) as f32 / d as f32);
                let (sin, cos) = freq.sin_cos();
                data[tok * n_heads * d + head * d + i] = a * cos - b * sin;
                data[tok * n_heads * d + head * d + i + d / 2] = b * cos + a * sin;
            }
        }
    }
}

// softmax(x) = exp(x - max) / sum(exp(x - max))
// y = softmax(mask(x))
pub fn masked_softmax(y: &mut Tensor<f32>) {
    let ndim = y.shape().len();
    assert!(ndim >= 2);
    let seq_len = y.shape()[ndim - 2];
    let total_seq_len = y.shape()[ndim - 1];
    let batch = y.size() / (seq_len * total_seq_len);
    let data = unsafe { y.data_mut() };
    for b in 0..batch {
        let base = b * seq_len * total_seq_len;
        for i in 0..seq_len {
            let offset = base + i * total_seq_len;
            let boundary = total_seq_len - seq_len + i + 1;

            let max = data[offset..offset + boundary]
                .iter()
                .fold(data[offset], |a, b| a.max(*b));

            let sum = (0..boundary)
                .map(|j| {
                    let e = (data[offset + j] - max).exp();
                    data[offset + j] = e;
                    e
                })
                .sum::<f32>();

            (0..boundary).for_each(|j| data[offset + j] /= sum);
            (boundary..total_seq_len).for_each(|j| data[offset + j] = 0.0);
        }
    }
}

pub fn rms_norm(y: &mut Tensor<f32>, x: &Tensor<f32>, w: &Tensor<f32>, epsilon: f32) {
    // let mut y = Tensor::<f32>::new(vec![1., 2., 3., 4.], &vec![2, 2]);
    // let x = Tensor::<f32>::new(vec![1., 2., 3., 4.], &vec![2, 2]);
    // let w = Tensor::<f32>::new(vec![1., 2.], &vec![2]);
    // rms_norm(&mut y, &x, &w, 1e-6);
    // assert!(y.close_to(
    //     &Tensor::<f32>::new(
    //         vec![0.6324554, 2.5298216, 0.8485281, 2.2627416],
    //         &vec![2, 2]
    //     ),
    //     1e-3
    // ));

    // x: 2*2, y: 2*2, w: 1*2

    // x的每行，都是长度为n的向量
    // y的每行，都是长度为n的向量

    // x01, x02, x03, …, x0n
    // x11, x12, x13, …, x1n
    // …

    // w 是一个一维向量， 长度为n（即和x的每行n 一样，和y的每行n也一样）
    // w1, w2, w3, …, wn

    // 进行 element-wise 乘法

    let x_len = x.size(); // x 是 n行n列
    let w_len = w.size(); // w 是 1行n列
    let x_lines_cnt = x_len / w_len; // x 共几行

    let mut _y = unsafe { y.data_mut() }; // y 的原生数组

    (0..x_lines_cnt).for_each(|i| {
        // i 为行号
        let x_one_line = x.slice(i * w_len, &vec![w_len]); // 取x的本行的切片(即n个元素, 其实是和w形状相同)
        let x_one_line_data = x_one_line.data(); // 从切片拿到实际的原生数组数据

        let sum_squares = x_one_line_data.iter().map(|x| x * x).sum::<f32>(); // 求x的此行, 的平方和, 是一个标量数字
        let sqrt = ((sum_squares / w_len as f32) + epsilon).sqrt(); // 求开方, 得到分母(即x的此行内各元素, 的此值相同)

        // 遍历本行x的每个元素, 算出本行y的每个元素
        (0..w_len).for_each(|j| {
            // 现在 w 是 1行n列, x_one_line_data 也是 1行n列
            // 对应元素相乘, 再除以 sqrt, 即得到 y 在此行各元素的值
            _y[i * w_len + j] = w.data()[j] * x_one_line_data[j] / sqrt;
        });
    })
}

fn close_to(x: f32, y: f32) -> bool {
    float_eq(&x, &y, 1e-3)
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

// y = sigmoid(x) * x * y
// hint: this is an element-wise operation
pub fn silu(y: &mut Tensor<f32>, x: &Tensor<f32>) {
    let len = y.size();
    assert!(len == x.size());

    let _y = unsafe { y.data_mut() };
    let _x = x.data();

    (0.._x.len()).for_each(|i| {
        _y[i] = sigmoid(_x[i]) * _x[i] * _y[i];
    })
}

// C = beta * C + alpha * A @ B^T
// hint: You don't need to do an explicit transpose of B
// 输入的bt, 是b的转置
pub fn matmul_transb(
    c: &mut Tensor<f32>,
    beta: f32,
    a: &Tensor<f32>,
    bt: &Tensor<f32>,
    alpha: f32,
) {
    // let mut c = Tensor::<f32>::new(vec![1., 2., 3., 4.], &vec![2, 2]);
    // let a = Tensor::<f32>::new(vec![1., 2., 3., 4., 5., 6.], &vec![2, 3]);
    // let b = Tensor::<f32>::new(vec![1., 2., 3., 4., 5., 6.], &vec![2, 3]);
    // matmul_transb(&mut c, 1., &a, &b, 1.);

    // 输出的c是 2*2, 输入的a是 2*3, 输入的bt是2*3(b是3*2)
    // 从而 a * bt = c 是2*2的
    assert_eq!(c.shape().len(), 2);
    assert_eq!(a.shape().len(), 2);
    assert_eq!(bt.shape().len(), 2);
    let x_num = c.shape()[0]; // 2
    let y_num = c.shape()[1]; // 2
    let k_num = a.shape()[1]; // 3
    let _c = unsafe { c.data_mut() };

    for x in 0..x_num {
        // x是a的第几行
        let row = &a.data()[x * k_num..(x + 1) * k_num]; // a 的 row
        for y in 0..y_num {
            // y是b的第几列
            let col = &bt.data()[y * k_num..(y + 1) * k_num]; // b 的 col(即bt的某行)
            let sum = row.iter().zip(col.iter()).map(|(a, b)| a * b).sum::<f32>();
            _c[x * y_num + y] = beta * _c[x * y_num + y] + alpha * sum;
        }
    }
}

pub fn add(x: &mut Tensor<f32>, y: &Tensor<f32>) {
    let len = x.size();
    assert_eq!(len, y.size());
    let mut _x = unsafe { x.data_mut() };
    (0..len).for_each(|i| _x[i] += y.data()[i])
}

// Dot product of two tensors (treated as vectors)
#[allow(unused)]
pub fn dot(x: &Tensor<f32>, y: &Tensor<f32>) -> f32 {
    let len = x.size();
    assert!(len == y.size());
    let x_ = x.data();
    let y_ = y.data();
    let mut sum = 0.0;
    for i in 0..len {
        sum += x_[i] * y_[i];
    }
    sum
}

// Sample a index from a tensor (treated as a probability vector)
pub fn random_sample(x: &Tensor<f32>, top_p: f32, top_k: u32, temperature: f32) -> u32 {
    assert!(x.shape()[x.shape().len() - 1] == x.size());
    if temperature <= 0. || top_k < 2 || top_p <= 0. {
        return x
            .data()
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap()
            .0 as _;
    }

    #[derive(Clone, Copy, PartialEq, Debug)]
    struct Probability {
        val: f32,
        tok: u32,
    }
    impl Eq for Probability {}
    impl PartialOrd for Probability {
        #[inline]
        fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
            Some(self.cmp(other))
        }
    }
    impl Ord for Probability {
        #[inline]
        fn cmp(&self, other: &Self) -> std::cmp::Ordering {
            match self.val.total_cmp(&other.val) {
                std::cmp::Ordering::Equal => self.tok.cmp(&other.tok),
                ord => ord.reverse(),
            }
        }
    }
    impl From<(usize, &f32)> for Probability {
        #[inline]
        fn from((i, p): (usize, &f32)) -> Self {
            Self {
                val: p.clone(),
                tok: i as _,
            }
        }
    }

    // sort
    let mut logits = x
        .data()
        .iter()
        .enumerate()
        .map(Probability::from)
        .collect::<Vec<_>>();
    logits.sort_unstable();
    let max = core::mem::replace(&mut logits[0].val, 1.);
    // softmax & sum
    for i in 1..logits.len() {
        logits[i].val = logits[i - 1].val + ((logits[i].val - max) / temperature).exp();
    }
    // topk & topp & random
    let pk = logits[(top_k as usize).min(logits.len()) - 1].val;
    let pp = logits[logits.len() - 1].val * top_p;
    let plimit = rand::random::<f32>() * f32::min(pk, pp);
    // sample
    logits.iter().find(|p| p.val >= plimit).unwrap().tok
}

#[test]
fn test_close_to() {
    assert!(close_to(0., 0.));
    assert!(close_to(1., 1.));
    assert!(close_to(-1., -1.));
}

#[test]
fn test_sigmoid() {
    let rel = 1e-3;
    float_eq(&sigmoid(0.), &0.5, rel);
    float_eq(&sigmoid(6.), &1.0, rel);
    float_eq(&sigmoid(100.), &1.0, rel);
    float_eq(&sigmoid(-6.), &-1., rel);
    float_eq(&sigmoid(-100.), &-1., rel);
}

// Your implementation should at least pass the following tests:
#[test]
fn test_silu() {
    let mut y = Tensor::<f32>::new(vec![2., 3., 4.], &vec![1, 3]);
    let x = Tensor::<f32>::new(vec![1., 2., 3.], &vec![1, 3]);
    silu(&mut y, &x);
    assert!(y.close_to(
        &Tensor::<f32>::new(vec![1.4621172, 5.2847824, 11.43089], &vec![1, 3]),
        1e-3
    ));
}

#[test]
fn test_rms_norm() {
    let mut y = Tensor::<f32>::new(vec![1., 2., 3., 4.], &vec![2, 2]);
    let x = Tensor::<f32>::new(vec![1., 2., 3., 4.], &vec![2, 2]);
    let w = Tensor::<f32>::new(vec![1., 2.], &vec![2]);
    rms_norm(&mut y, &x, &w, 1e-6);
    assert!(y.close_to(
        &Tensor::<f32>::new(
            vec![0.6324554, 2.5298216, 0.8485281, 2.2627416],
            &vec![2, 2]
        ),
        1e-3
    ));
}

#[test]
fn test_matmul_transb() {
    let mut c = Tensor::<f32>::new(vec![1., 2., 3., 4.], &vec![2, 2]);
    let a = Tensor::<f32>::new(vec![1., 2., 3., 4., 5., 6.], &vec![2, 3]);
    let b = Tensor::<f32>::new(vec![1., 2., 3., 4., 5., 6.], &vec![2, 3]);
    matmul_transb(&mut c, 1., &a, &b, 1.);
    assert!(c.close_to(
        &Tensor::<f32>::new(vec![15., 34., 35., 81.], &vec![2, 2]),
        1e-3
    ));
}

#[test]
fn test_add() {
    let mut a = Tensor::<f32>::new(vec![1., 2., 3., 4.], &vec![2, 2]);
    let b = Tensor::<f32>::new(vec![5., 6., 7., 8.], &vec![2, 2]);
    add(&mut a, &b);
    assert_eq!(a.size(), 4);
    assert!(a.close_to(
        &Tensor::<f32>::new(vec![6., 8., 10., 12.], &vec![2, 2]),
        1e-3
    ));
}
