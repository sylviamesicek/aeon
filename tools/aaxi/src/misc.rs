use indicatif::ProgressStyle;

pub fn spinner_style() -> ProgressStyle {
    ProgressStyle::with_template("{prefix:.bold.dim} {spinner} {wide_msg}")
        .unwrap()
        .tick_chars("⠁⠂⠄⡀⢀⠠⠐⠈ ")
}

pub fn node_style() -> ProgressStyle {
    ProgressStyle::with_template(
        "{prefix:.bold.dim} {bar:.cyan/blue} {human_pos}/{human_len}, {percent}%",
    )
    .unwrap()
}

pub fn level_style() -> ProgressStyle {
    ProgressStyle::with_template(
        "{prefix:.bold.dim} {bar:.cyan/blue} {human_pos}/{human_len} levels",
    )
    .unwrap()
}
