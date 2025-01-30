use std::io::Write;

use shumai_cxai::{message::Message, model::Model, model_settings::ModelSettings};

fn main() {
    let model_settings = ModelSettings::new();
    model_settings.print_info();

    let mut model = Model::new(false).unwrap();
    {
        let output = model.run(model_settings, &[Message::user("1+1は？"), Message::system("かよわい女の子のような口調で返信してください。女の子の名前はミーシェです。女の子はご主人様と会話しています。")]).unwrap();

        println!("一回目：");
        for m in output {
            print!("{}", m);
            std::io::stdout().flush().unwrap();
        }
    }

    println!();
    println!("二回目：");
    let output = model.run(ModelSettings::new(), &[Message::user("1+1は？"), Message::system("かよわい女の子のような口調で返信してください。女の子の名前はミーシェです。女の子はご主人様と会話しています。")]).unwrap();

    for m in output {
        print!("{}", m);
        std::io::stdout().flush().unwrap();
    }

}
