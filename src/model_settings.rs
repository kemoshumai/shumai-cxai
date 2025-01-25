pub struct ModelSettings {
    pub temperature: f64,
    pub repeat_penalty: f32,
    pub repeat_last_n: usize,
    pub k: usize,
    pub p: f64,
    pub seed: u64,
    pub sample_len: u64,
}

impl Default for ModelSettings {
    fn default() -> Self {
        Self {
            temperature: 0.8,
            repeat_penalty: 1.1,
            repeat_last_n: 10,
            k: 40,
            p: 0.95,
            seed: 42,
            sample_len: 4096u64,
        }
    }
}

impl ModelSettings {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn print_info(&self) {
        println!(
            "avx: {}, neon: {}, simd128: {}, f16c: {}",
            candle_core::utils::with_avx(),
            candle_core::utils::with_neon(),
            candle_core::utils::with_simd128(),
            candle_core::utils::with_f16c()
        );
        println!(
            "temp: {:.2} repeat-penalty: {:.2} repeat-last-n: {}",
            self.temperature, self.repeat_penalty, self.repeat_last_n
        );
    }
}