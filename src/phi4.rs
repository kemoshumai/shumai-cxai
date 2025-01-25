use std::io::Write;
use candle_examples::token_output_stream::TokenOutputStream;
use tokenizers::Tokenizer;

use candle_core::quantized::gguf_file;
use candle_core::Tensor;
use candle_transformers::generation::{LogitsProcessor, Sampling};

use candle_transformers::models::quantized_phi3::ModelWeights as Phi3;

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

pub fn main() -> anyhow::Result<()> {

    let temperature = 0.8;
    let repeat_penalty = 1.1;
    let repeat_last_n = 10;
    let k = 40;
    let p = 0.95;

    let flash_attn = false;
    let seed = 42;
    let sample_len = 100u64;

    let cpu = false;

    println!(
        "avx: {}, neon: {}, simd128: {}, f16c: {}",
        candle_core::utils::with_avx(),
        candle_core::utils::with_neon(),
        candle_core::utils::with_simd128(),
        candle_core::utils::with_f16c()
    );
    println!(
        "temp: {:.2} repeat-penalty: {:.2} repeat-last-n: {}",
        temperature, repeat_penalty, repeat_last_n
    );
    

    let model_path = hf_hub::api::sync::Api::new()?
        .repo(hf_hub::Repo::with_revision("microsoft/phi-4-gguf".to_string(),hf_hub::RepoType::Model,"main".to_string()))
        .get("phi-4-q4.gguf")?
        ;

    let mut file = std::fs::File::open(&model_path)?;
    let start = std::time::Instant::now();
    let device = candle_examples::device(cpu)?;

    let mut model = {
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

    let tokenizer = Tokenizer::from_file(hf_hub::api::sync::Api::new()?.model("microsoft/phi-4".to_string()).get("tokenizer.json")?).unwrap();
    let mut tos = TokenOutputStream::new(tokenizer);

    let prompt_str = "こんにちは！1+1は田んぼの田ですか？";
    println!("prompt_str: {}", &prompt_str);

    let tokens = tos
        .tokenizer()
        .encode(prompt_str, true)
        .map_err(anyhow::Error::msg)?;
    let tokens = tokens.get_ids();
    let to_sample = sample_len.saturating_sub(1);
    let mut logits_processor = LogitsProcessor::from_sampling(seed, Sampling::TopKThenTopP { k, p, temperature } );

    println!("logits_processor built");

    println!("tokens: {:?}", tokens);

    let mut next_token =  {
        let input = Tensor::new(tokens, &device)?.unsqueeze(0)?;
        let logits = model.forward(&input, 0)?;
        let logits = logits.squeeze(0)?;
        logits_processor.sample(&logits)?
    };

    let mut all_tokens = vec![];
    all_tokens.push(next_token);
    if let Some(t) = tos.next_token(next_token)? {
        print!("{t}");
        std::io::stdout().flush()?;// ころっけに醤油をかける園田智代子
    }

    let eos_token = *tos
        .tokenizer()
        .get_vocab(true)
        .get("<|endoftext|>")
        .unwrap();

    let start_post_prompt = std::time::Instant::now();
    let mut sampled = 0;
    for index in 0..to_sample {
        let input = Tensor::new(&[next_token], &device)?.unsqueeze(0)?;
        let logits = model.forward(&input, tokens.len() + index as usize)?;
        let logits = logits.squeeze(0)?;
        let logits = if repeat_penalty == 1. {
            logits
        } else {
            let start_at = all_tokens.len().saturating_sub(repeat_last_n);
            candle_transformers::utils::apply_repeat_penalty(
                &logits,
                repeat_penalty,
                &all_tokens[start_at..],
            )?
        };
        next_token = logits_processor.sample(&logits)?;
        all_tokens.push(next_token);
        if let Some(t) = tos.next_token(next_token)? {
            print!("{t}");
            std::io::stdout().flush()?;
        }
        sampled += 1;
        if next_token == eos_token {
            break;
        };
    }


    if let Some(rest) = tos.decode_rest().map_err(candle_core::Error::msg)? {
        print!("{rest}");
    }

    std::io::stdout().flush()?;
    let dt = start_post_prompt.elapsed();
    println!();
    println!(
        "{sampled:4} tokens generated: {:.2} token/s",
        sampled as f64 / dt.as_secs_f64(),
    );

    Ok(())
}