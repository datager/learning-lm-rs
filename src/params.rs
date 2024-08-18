use std::vec;

use crate::config::LlamaConfigJson;
use crate::tensor::Tensor;
use safetensors::SafeTensors;
pub struct LLamaParams<T> {
    // token_id to embedding lookup table
    pub embedding_table: Tensor<T>, // (vocab_size, dim)
    // decoder layer
    pub rms_att_w: Vec<Tensor<T>>, // (hidden_size, ) x layers
    pub wq: Vec<Tensor<T>>,        // (n_heads * head_size, hidden_size) x layers
    pub wk: Vec<Tensor<T>>,        // (n_kv_heads * head_size, hidden_size) x layers
    pub wv: Vec<Tensor<T>>,        // (n_kv_heads * head_size, hidden_size) x layers
    pub wo: Vec<Tensor<T>>,        // (hidden_size, n_heads * head_size) x layers
    // ffn layer
    pub rms_ffn_w: Vec<Tensor<T>>, // (hidden_size, ) x layers
    pub w_up: Vec<Tensor<T>>,      // (intermediate_size, hidden_size) x layers
    pub w_gate: Vec<Tensor<T>>,    // (intermediate_size, hidden_size) x layers
    pub w_down: Vec<Tensor<T>>,    // (hidden_size, intermediate_size) x layers
    // output
    pub rms_out_w: Tensor<T>, // (hidden_size, )
    pub lm_head: Tensor<T>,   // (vocab_size, dim)
}

impl LLamaParams<f32> {
    // 加载 hugging face 官方的 satetensor 结构, 从其中的 metadata 元信息中得到各矩阵的参数, 初始化本项目自定义的 LLamaParam 结构体
    pub fn from_safetensors(safetensor: &SafeTensors, config: &LlamaConfigJson) -> Self {
        let layers = config.num_hidden_layers; // 共 2 层

        // 从safetensors文件的模型参数加载
        // let get_tensor = |name: &str| {
        //     let tensor_view = safetensor.tensor(name).unwrap(); // 断点可看到 safetensor加载出的内容, data 是数据, metadata是元数据, 我们根据 name 可以先在 metadata.index_map 中找到 metadata.tensors的下标, 再在 metadata.tensors 中找到data[] 中的下标, 最终在 data[] 中按下标找到对应范围的元素
        //     let mut vec_f32s = vec![];
        //     for chunk in tensor_view.data().chunks_exact(4) {
        //         // 因为参数是 float32 的, 1个 float32 占用 4 个 bytes
        //         // 每次迭代取4个bytes
        //         let bytes: [u8; 4] = chunk.try_into().unwrap(); // 取4个bytes
        //         let f = f32::from_le_bytes(bytes); // 把这4个bytes 转为 1个float32
        //         vec_f32s.push(f);
        //     }
        //     Tensor::new(vec_f32s, &tensor_view.shape().to_vec()) // Vec<f32> 转 Tensor<f32>, 其中 Tensor 是本项目自定义的结构, 这样就实现了把 huggingface 通用的 safetensor 文件中的参数, 转换为本项目自定义的 Tensor 结构了
        // };

        fn get_tensor(name: &str, safetensor: &SafeTensors) -> Tensor<f32> {
            let tensor_view = safetensor.tensor(name).unwrap(); // 断点可看到 safetensor加载出的内容, data 是数据, metadata是元数据, 我们根据 name 可以先在 metadata.index_map 中找到 metadata.tensors的下标, 再在 metadata.tensors 中找到data[] 中的下标, 最终在 data[] 中按下标找到对应范围的元素
            let mut vec_f32s = vec![];
            for chunk in tensor_view.data().chunks_exact(4) {
                // 因为参数是 float32 的, 1个 float32 占用 4 个 bytes
                // 每次迭代取4个bytes
                let bytes: [u8; 4] = chunk.try_into().unwrap(); // 取4个bytes
                let f = f32::from_le_bytes(bytes); // 把这4个bytes 转为 1个float32
                vec_f32s.push(f);
            }
            Tensor::new(vec_f32s, &tensor_view.shape().to_vec()) // Vec<f32> 转 Tensor<f32>, 其中 Tensor 是本项目自定义的结构, 这样就实现了把 huggingface 通用的 safetensor 文件中的参数, 转换为本项目自定义的 Tensor 结构了
        }
        let f_get_tensor = |name: &str| get_tensor(name, safetensor);

        fn get_tensor_vec(safetensor: &SafeTensors, layers: usize, name: &str) -> Vec<Tensor<f32>> {
            let mut tensors = vec![];
            for layer_idx in 0..layers {
                // 如 model.layers.1.self_attn.o_proj.weight
                // 如 model.layers.0.mlp.gate_proj.weight
                // 如 model.layers.1.mlp.down_proj.weight
                let name = format!("model.layers.{layer_idx}.{name}");
                let tensor = get_tensor(&name, safetensor);
                tensors.push(tensor);
            }
            tensors
        }
        let f_get_tensor_vec =
            |name: &str| -> Vec<Tensor<f32>> { get_tensor_vec(safetensor, layers, name) };

        LLamaParams {
            // 结构为 Tensor<f32>
            embedding_table: f_get_tensor("lm_head.weight"), // 维度是 (2048, 128)
            rms_out_w: f_get_tensor("model.norm.weight"),    // 维度是 (1, 128)
            lm_head: f_get_tensor("lm_head.weight"), // 和 embedding_table 可以相同, 取决于 tie_word_embeddings 配置项

            // 结构为 Vec<Tensor<f32>>
            rms_att_w: f_get_tensor_vec("input_layernorm.weight"),
            rms_ffn_w: f_get_tensor_vec("post_attention_layernorm.weight"),
            wq: f_get_tensor_vec("self_attn.q_proj.weight"),
            wk: f_get_tensor_vec("self_attn.k_proj.weight"),
            wv: f_get_tensor_vec("self_attn.v_proj.weight"),
            wo: f_get_tensor_vec("self_attn.o_proj.weight"),
            w_up: f_get_tensor_vec("mlp.up_proj.weight"),
            w_gate: f_get_tensor_vec("mlp.gate_proj.weight"),
            w_down: f_get_tensor_vec("mlp.down_proj.weight"),
        }
    }
}
