
use candle_examples::token_output_stream::TokenOutputStream;
use candle_transformers::{generation::{LogitsProcessor, Sampling}, models::quantized_phi3::ModelWeights};

use candle_core::{quantized::gguf_file, Tensor};

use candle_transformers::models::quantized_phi3::ModelWeights as Phi3;
use tokenizers::Tokenizer;

use crate::{message::Message, model_settings::ModelSettings};

fn format_size(size_in_bytes: usize) -> String {
    if size_in_bytes < 1_000 {
        format!("{}B", size_in_bytes)
    } else if size_in_bytes < 1_000_000 {
        format!("{:.2}KB", size_in_bytes as f64 / 1e3)
    } else if size_in_bytes < 1_000_000_000 {
        format!("{:.2}MB", size_in_bytes as f64 / 1e6)
    } else {
        format!("{:.2}GB", size_in_bytes as f64 / 1e9)
    }
}


pub struct Model {
    device: candle_core::Device,
    model: ModelWeights,
}

impl Model {
    pub fn new(flash_attn: bool, cpu: bool) -> anyhow::Result<Self> {

        let model_path = hf_hub::api::sync::Api::new()?
            .repo(hf_hub::Repo::with_revision("microsoft/phi-4-gguf".to_string(),hf_hub::RepoType::Model,"main".to_string()))
            .get("phi-4-q4.gguf")?
            ;

        let mut file = std::fs::File::open(&model_path)?;
        let start = std::time::Instant::now();
        let device = candle_examples::device(cpu)?;

        let model = {
            let model = gguf_file::Content::read(&mut file).map_err(|e| e.with_path(model_path))?;
            let mut total_size_in_bytes = 0;
            for (_, tensor) in model.tensor_infos.iter() {
                let elem_count = tensor.shape.elem_count();
                total_size_in_bytes +=
                    elem_count * tensor.ggml_dtype.type_size() / tensor.ggml_dtype.block_size();
            }
            println!(
                "loaded {:?} tensors ({}) in {:.2}s",
                model.tensor_infos.len(),
                &format_size(total_size_in_bytes),
                start.elapsed().as_secs_f32(),
            );
            Phi3::from_gguf(
                flash_attn,
                model,
                &mut file,
                &device,
            )?
        };
        println!("model built");

        Ok(Self {
            device,
            model,
        })
    }

    pub fn run(&mut self, model_settings: ModelSettings, messages: &[Message]) -> anyhow::Result<ModelOutput> {
        let ModelSettings {
            temperature,
            repeat_penalty,
            repeat_last_n,
            k,
            p,
            seed,
            sample_len,
        } = model_settings;

        let model = &mut self.model;
        let device = &mut self.device;

        // トークナイザーを用意する
        let tokenizer = Tokenizer::from_file(hf_hub::api::sync::Api::new()?.model("microsoft/phi-4".to_string()).get("tokenizer.json")?).unwrap();
        let tos = TokenOutputStream::new(tokenizer);

        // メッセージを結合してプロンプト文字列を作成
        let prompt_str = messages.iter().map(Message::to_string).collect::<String>() + "<|im_start|>assistant<|im_sep|>";

        // プロンプト文字列をトークン化
        let tokens = tos
            .tokenizer()
            .encode(prompt_str, true)
            .map_err(anyhow::Error::msg)?;
        let tokens = tokens.get_ids();
        let to_sample = sample_len.saturating_sub(1);
        let mut logits_processor = LogitsProcessor::from_sampling(seed, Sampling::TopKThenTopP { k, p, temperature } );

        // 最初のトークンをサンプリング
        let next_token =  {
            let input = Tensor::new(tokens, device)?.unsqueeze(0)?;
            let logits = model.forward(&input, 0)?;
            let logits = logits.squeeze(0)?;
            logits_processor.sample(&logits)?
        };

        Ok(ModelOutput {
            model,
            to_sample: to_sample as usize,
            eos_token: 0,
            repeat_penalty,
            repeat_last_n,
            all_tokens: Vec::new(),
            next_token,
            device,
            tokens: tokens.to_vec(),
            logits_processor,
            tos,
            sampled: 0,
            index: 0,
            initial: true,
            eos: false,
        })
    }
}

pub struct ModelOutput<'a> {
    model: &'a mut ModelWeights,
    to_sample: usize,
    eos_token: u32,
    repeat_penalty: f32,
    repeat_last_n: usize,
    all_tokens: Vec<u32>,
    next_token: u32,
    device: &'a mut candle_core::Device,
    tokens: Vec<u32>,
    logits_processor: LogitsProcessor,
    tos: TokenOutputStream,
    sampled: u64,// init this to 0
    index: usize,
    initial: bool,
    eos: bool,
}

impl<'a> Iterator for ModelOutput<'a> {
    type Item = String;

    fn next(&mut self) -> Option<Self::Item> {

        // 初回実行時の処理
        if self.initial {
            self.initial = false;

            self.all_tokens = [self.next_token].to_vec();

            let t = self.tos.next_token(self.next_token).unwrap();

            // EOSトークンを取得
            self.eos_token = *self.tos
                .tokenizer()
                .get_vocab(true)
                .get("<|endoftext|>")
                .unwrap();

            // 初期化
            self.sampled = 0;
            self.index = 0;

            return t;
        }

        if self.index < self.to_sample && !self.eos {

            let mut f = || {
                let input = Tensor::new(&[self.next_token], self.device).unwrap().unsqueeze(0).unwrap();
                let logits = self.model.forward(&input, self.tokens.len() + self.index).unwrap();
                self.index += 1;// indexを参照後にインクリメント
                let logits = logits.squeeze(0).unwrap();
                let logits = if self.repeat_penalty == 1. {
                    logits
                } else {
                    let start_at = self.all_tokens.len().saturating_sub(self.repeat_last_n);
                    candle_transformers::utils::apply_repeat_penalty(
                        &logits,
                        self.repeat_penalty,
                        &self.all_tokens[start_at..],
                    ).unwrap()
                };
                self.next_token = self.logits_processor.sample(&logits).unwrap();
                self.all_tokens.push(self.next_token);

                self.sampled += 1;

                if self.next_token == self.eos_token {
                    self.eos = true;
                }

                self.tos.next_token(self.next_token).unwrap()
            };

            let mut t = f();

            while t.is_none() {
                t = f();
            }

            if let Some(t) = t {
                return Some(t);
            }

        }

        self.tos.decode_rest().map_err(candle_core::Error::msg).unwrap()

    }
}