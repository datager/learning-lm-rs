use serde;
#[derive(serde::Serialize, serde::Deserialize, Debug)]
pub(crate) struct LlamaConfigJson {
    pub bos_token_id: u32,              // 起始符 token id = 1
    pub eos_token_id: u32,              // 结束符 token id = 2
    pub hidden_size: usize,             // 隐藏层大小, 即各层输出的最后一维 = 128
    pub intermediate_size: usize,       // feed forward 神经网络中间层的大小 = 384
    pub max_position_embeddings: usize, // 最大序列长度 = 512
    pub num_attention_heads: usize,     // self-attention 的 Q 头数 = 8
    pub num_hidden_layers: usize,       // 隐藏层数 = 2
    pub num_key_value_heads: usize,     // self-attention 的 K和V的 头数 = 8
    pub vocab_size: usize,              // 词表大小 = 2048
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f32, // RMS Norm 层的 epsilon参数 = 1e-6
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32, // RoPE的theta = 10000
    pub torch_dtype: String,            // 模型数据类型 = 'float32'
    #[serde(default = "default_tie_word_embeddings")]
    pub tie_word_embeddings: bool, // 起始和结束embedding参数矩阵, 是否共享同一份数据 = true
}

#[inline(always)]
const fn default_rms_norm_eps() -> f32 {
    1e-5
}

#[inline(always)]
const fn default_rope_theta() -> f32 {
    1e4
}

#[inline(always)]
const fn default_tie_word_embeddings() -> bool {
    false
}
