use shumai_cxai::{message::Message, model::Model, model_settings::ModelSettings};

fn main() {
    let model_settings = ModelSettings::new();
    model_settings.print_info();

    let mut model = Model::new(false, false).unwrap();
    model.run(model_settings, &[Message::user("こんにちは！"), Message::system("かよわい女の子のような口調で返信してください。女の子の名前はミーシェです。女の子はご主人様と会話しています。")]).unwrap();

}
