//! Handles applying configuration variables/arguments to a config struct.

use chumsky::{extra::Err, prelude::*, text::digits};
use std::collections::HashMap;
use std::fmt::Write as _;
use thiserror::Error;

/// Arguments in scope when transforming config strings.
#[derive(Clone, Debug)]
pub struct ConfigVars {
    /// Positional arguments passed into cli invokation.
    pub positional: Vec<f64>,
    /// Named arguments provided by search.
    pub named: HashMap<String, f64>,
}

impl ConfigVars {
    pub fn new() -> Self {
        Self {
            positional: Vec::new(),
            named: HashMap::new(),
        }
    }
}

impl Default for ConfigVars {
    fn default() -> Self {
        Self::new()
    }
}

/// Error while parsing and transforming strings in a struct.
#[derive(Error, Debug)]
pub enum TransformError {
    #[error("positional argument {0} out of bounds for arguments length {1}")]
    PositionalOutOfBounds(usize, usize),
    #[error("named argument {0} does not exist")]
    NameDoesNotExist(String),
    #[error("failed to parse config string as sequence of tokens: {0}")]
    TokenParseFailed(String),
    #[error("string format failed")]
    FormatFailed(#[from] std::fmt::Error),
    #[error("failed to parse string as float: {0}")]
    FloatParseFailed(#[from] std::num::ParseFloatError),
    #[error("failed to parse string as usize: {0}")]
    IntParseFailed(#[from] std::num::ParseIntError),
}

/// Trait for applying config variable transformation for a struct.
pub trait Transform {
    type Output;
    /// Transforms a given input by parsing and substituting any patterns of the form
    /// "$digit", "${number}", "${string}". The former two map to positional arguments
    /// and the latter references named arguments.
    fn transform(&self, vars: &ConfigVars) -> Result<Self::Output, TransformError>;
}

impl Transform for str {
    type Output = String;

    fn transform(&self, vars: &ConfigVars) -> Result<Self::Output, TransformError> {
        let parser = token_stream_parser();
        // TODO properly propogate errors
        let tokens = parser.parse(self).into_result().map_err(|err| {
            assert!(err.len() > 0);

            return TransformError::TokenParseFailed(self.to_string());
        })?;

        let mut result = String::new();

        for token in tokens {
            match token {
                Token::String(string) => result.push_str(string),
                Token::Positional(idx) => {
                    let Some(name) = vars.positional.get(idx) else {
                        return Err(TransformError::PositionalOutOfBounds(
                            idx,
                            vars.positional.len(),
                        ));
                    };
                    write!(result, "{}", name)?;
                }
                Token::Named(key) => {
                    let Some(name) = vars.named.get(key) else {
                        return Err(TransformError::NameDoesNotExist(key.to_string()));
                    };
                    write!(result, "{}", name)?;
                }
            }
        }

        Ok(result)
    }
}

impl<T: Transform> Transform for Vec<T> {
    type Output = Vec<T::Output>;

    fn transform(&self, vars: &ConfigVars) -> Result<Vec<T::Output>, TransformError> {
        self.iter()
            .map(|source| {
                let v = source.transform(vars);
                v
            })
            .collect::<_>()
    }
}

/// A floating point argument that can either be provided via a configuration file
/// or as a config variable.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(untagged)]
pub enum FloatVar {
    /// Fixed floating point input.
    Inline(f64),
    /// Script that will be parsed by the transformer
    Script(String),
}

impl FloatVar {
    /// Unwraps a float var into a float, assuming that it has already been transformed.
    pub fn unwrap(&self) -> f64 {
        let Self::Inline(v) = self else {
            panic!("failed to unwrap FloatVar");
        };

        *v
    }

    pub fn is_transformed(&self) -> bool {
        matches!(self, FloatVar::Inline(_))
    }
}

impl From<f64> for FloatVar {
    fn from(value: f64) -> Self {
        Self::Inline(value)
    }
}

impl Transform for FloatVar {
    type Output = Self;

    fn transform(&self, vars: &ConfigVars) -> Result<Self::Output, TransformError> {
        Ok(FloatVar::Inline(match self {
            FloatVar::Inline(v) => *v,
            FloatVar::Script(pos) => pos.transform(vars)?.parse::<f64>()?,
        }))
    }
}

/// A unsigned integer argument that can either be provided via a configuration file
/// or as a config variable.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(untagged)]
pub enum UnsignedVar {
    /// Fixed floating point input.
    Inline(usize),
    /// Script that will be parsed by the transformer
    Script(String),
}

impl UnsignedVar {
    /// Unwraps a float var into a float, assuming that it has already been transformed.
    pub fn unwrap(&self) -> usize {
        let Self::Inline(v) = self else {
            panic!("failed to unwrap UnsignedVar");
        };

        *v
    }

    pub fn is_transformed(&self) -> bool {
        matches!(self, UnsignedVar::Inline(_))
    }
}

impl From<usize> for UnsignedVar {
    fn from(value: usize) -> Self {
        Self::Inline(value)
    }
}

impl Transform for UnsignedVar {
    type Output = Self;

    fn transform(&self, vars: &ConfigVars) -> Result<Self::Output, TransformError> {
        Ok(UnsignedVar::Inline(match self {
            UnsignedVar::Inline(v) => *v,
            UnsignedVar::Script(pos) => pos.transform(vars)?.parse::<usize>()?,
        }))
    }
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

type SimpleError<'a> = Err<Simple<'a, char>>;

/// Parser of a string into a token.
fn token_parser<'a>() -> impl Parser<'a, &'a str, Token<'a>, SimpleError<'a>> {
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
            "some/${arg1}".transform(&ctx).unwrap(),
            "some/10".to_string()
        );

        assert_eq!(
            "$0$1${02}${0h@!!}".transform(&ctx).unwrap(),
            "01.12.211".to_string()
        );
    }
}
