use hoplite::uci::Uci;

#[test]
fn go_without_nnue_reports_error() {
    let mut uci = Uci::new();
    let mut output = Vec::new();
    uci.handle_command("go", &mut output);
    let text = String::from_utf8(output).expect("uci output should be utf8");
    assert!(
        text.contains(
            "info string error cannot start search: no NNUE evaluator is currently loaded"
        ),
        "expected error message when starting search without NNUE, got: {}",
        text
    );
}
