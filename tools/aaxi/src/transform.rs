//! Handles applying configuration variables/arguments to a config struct.

use chumsky::{prelude::*, text::digits};
use eyre::eyre;
use std::collections::HashMap;
use std::fmt::Write as _;

/// Arguments in scope when transforming config strings.
#[derive(Clone, Debug)]
pub struct ConfigVars {
    /// Positional arguments passed into cli invokation.
    pub positional: Vec<f64>,
    /// Named arguments provided by search.
    pub named: HashMap<String, f64>,
}

/// Transforms a given input by parsing and substituting any patterns of the form
/// "$digit", "${number}", "${string}". The former two map to positional arguments
/// and the latter references named arguments.
pub fn transform<'a>(input: &'a str, vars: &ConfigVars) -> eyre::Result<String> {
    let parser = token_stream_parser();
    // TODO properly propogate errors
    let tokens = parser.parse(input).into_result().map_err(|err| {
        assert!(err.len() > 0);

        return eyre!("Parsing string failed");
    })?;

    let mut result = String::new();

    for token in tokens {
        match token {
            Token::String(string) => result.push_str(string),
            Token::Positional(idx) => {
                let Some(name) = vars.positional.get(idx) else {
                    return Err(eyre!(
                        "found index {} while parsing config file, but only {} positional arguments were provided",
                        idx,
                        vars.positional.len()
                    ));
                };
                write!(result, "{}", name)?;
            }
            Token::Named(key) => {
                let Some(name) = vars.named.get(key) else {
                    return Err(eyre!("named argument {} could not be found", key));
                };
                write!(result, "{}", name)?;
            }
        }
    }

    Ok(result)
}

/// A single token taken after parsing a string in the config file. The full string
/// can be formed by concating all tokens.
#[derive(Debug, PartialEq, Eq)]
enum Token<'src> {
    /// A unmatched string sequence without any `$` characters.
    String(&'src str),
    /// A positional argument of the form `$0` (only allowed if single digit) or `${0}`.
    Positional(usize),
    /// Named argument of the form `${name}`.
    Named(&'src str),
}

// type SimpleError<'a> = Err<Simple<'a, char>>;

/// Parser of a string into a token.
fn token_parser<'a>() -> impl Parser<'a, &'a str, Token<'a>> {
    let single_digit = one_of::<_, &str, _>("0123456789").map(|r| {
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

/// Parser for a input stream of strings into a set of tokens. It does this by
/// repeated applying `token_parser()`.
fn token_stream_parser<'a>() -> impl Parser<'a, &'a str, Vec<Token<'a>>> {
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

    #[test]
    fn basic_transform() {
        // Set up context
        let mut positional = Vec::new();
        positional.push(0.0);
        positional.push(1.1);
        positional.push(2.2);
        let mut named = HashMap::new();
        named.insert("arg1".to_string(), 10.0);
        named.insert("0h@!!".to_string(), 11.0);
        let ctx = ConfigVars { positional, named };

        assert_eq!(
            transform("some/${arg1}", &ctx).unwrap(),
            "some/10".to_string()
        );

        assert_eq!(
            transform("$0$1${02}${0h@!!}", &ctx).unwrap(),
            "01.12.211".to_string()
        );
    }
}
