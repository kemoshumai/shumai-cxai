use shumai_cxai::{model::Model, model_settings::ModelSettings};

fn main() {
    let model_settings = ModelSettings::new();
    model_settings.print_info();

    let mut model = Model::new(true, false).unwrap();
    model.run(model_settings).unwrap();

}
