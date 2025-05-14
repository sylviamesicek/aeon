use chumsky::{extra::Err, prelude::*, text::digits};
use std::collections::HashMap;

pub struct TransformContext {
    positional: Vec<f64>,
    named: HashMap<String, f64>,
}

#[derive(Debug, PartialEq, Eq)]
pub enum Token<'src> {
    String(&'src str),
    Positional(usize),
    Named(&'src str),
}

type SimpleError<'a> = Err<Simple<'a, char>>;

fn token_parser<'a>() -> impl Parser<'a, &'a str, Token<'a>, SimpleError<'a>> {
    let single_digit = one_of::<_, &str, SimpleError>("0123456789").map(|r| {
        let mut buffer = [0; 4];
        let slice = r.encode_utf8(&mut buffer);
        usize::from_str_radix(slice, 10).unwrap()
    });

    let singledigit_positional = just('$')
        .ignore_then(single_digit)
        .map(|v| Token::Positional(v));

    let multidigit_positional = just('$')
        .ignore_then(digits(10).to_slice().delimited_by(just('{'), just('}')))
        .map(|r| Token::Positional(usize::from_str_radix(r, 10).unwrap()));

    let multidigit_named = just('$')
        .ignore_then(
            none_of('}')
                .repeated()
                .to_slice()
                .delimited_by(just('{'), just('}')),
        )
        .map(|r| Token::Named(r));

    // let string = none_of('$').repeated().to_slice().map(|r| Token::String(r));

    let misc_string = none_of('$')
        .repeated()
        .at_least(1)
        .to_slice()
        .map(|r| Token::String(r));

    let token = choice((
        singledigit_positional,
        multidigit_positional,
        multidigit_named,
        misc_string,
    ));

    token
}

fn token_stream_parser<'a>() -> impl Parser<'a, &'a str, Vec<Token<'a>>, SimpleError<'a>> {
    token_parser()
        .repeated()
        .at_least(1)
        .collect::<Vec<Token>>()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn token_parsing() {
        let token = token_parser();

        assert_eq!(token.parse("$3").into_result(), Ok(Token::Positional(3)));
        assert_eq!(
            token.parse("${100}").into_result(),
            Ok(Token::Positional(100))
        );
        assert_eq!(
            token.parse("${1hello}").into_result(),
            Ok(Token::Named("1hello"))
        );
        assert_eq!(
            token.parse("Hello").into_result(),
            Ok(Token::String("Hello"))
        );
    }

    #[test]
    fn token_stream_parsing() {
        let tokens = token_stream_parser();

        assert_eq!(
            tokens.parse("hello$1${name}").into_result(),
            Ok(vec![
                Token::String("hello"),
                Token::Positional(1),
                Token::Named("name")
            ])
        );
        assert_eq!(
            tokens.parse("$100na${}").into_result(),
            Ok(vec![
                Token::Positional(1),
                Token::String("00na"),
                Token::Named("")
            ])
        );
    }
}
