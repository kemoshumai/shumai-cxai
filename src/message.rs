pub enum Message{
    User(String),
    Assistant(String),
    System(String),
}

impl Message {
    pub fn user<S: Into<String>>(s: S) -> Self {
        Self::User(s.into())
    }

    pub fn assistant<S: Into<String>>(s: S) -> Self {
        Self::Assistant(s.into())
    }

    pub fn system<S: Into<String>>(s: S) -> Self {
        Self::System(s.into())
    }
}

impl std::fmt::Display for Message {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::User(s) => write!(f, "<|im_start|>user<|im_sep|>\n{}\n<|im_end|>", s),
            Self::Assistant(s) => write!(f, "<|im_start|>assistant<|im_sep|>\n{}\n<|im_end|>", s),
            Self::System(s) => write!(f, "<|im_start|>system<|im_sep|>\n{}\n<|im_end|>", s),
        }
    }
}

impl From<&Message> for String {
    fn from(m: &Message) -> Self {
        m.to_string()
    }
}

impl From<Message> for String {
    fn from(m: Message) -> Self {
        m.to_string()
    }
}

#[test]
fn test_message() {
    let user = Message::user("おはよー！");
    let assistant = Message::assistant("おはよー！");
    let system = Message::system("おはよー！");

    assert_eq!(user.to_string(), "<|im_start|>user<|im_sep|>\nおはよー！\n<|im_end|>");
    assert_eq!(assistant.to_string(), "<|im_start|>assistant<|im_sep|>\nおはよー！\n<|im_end|>");
    assert_eq!(system.to_string(), "<|im_start|>system<|im_sep|>\nおはよー！\n<|im_end|>");
}