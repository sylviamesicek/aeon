use chumsky::{extra::Err, prelude::*};
use std::collections::HashMap;
use std::fmt::Write as _;
use thiserror::Error;

/// Simple error used in chumsky parsers.
type SimpleError<'a> = Err<Simple<'a, char>>;

/// Error while parsing and transforming strings in a struct.
#[derive(Error, Debug, PartialEq, Eq)]
pub enum VarDefParseError {
    #[error("parsing variable definition threw error: {0}")]
    ParseFailed(String),
}

/// A variable definition of the form "<KEY>=<VALUE>".
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VarDef<'a> {
    /// Name of the variable (referenced via ${<KEY>}).
    pub key: &'a str,
    /// Value of the variable (replaces ${<KEY>} on parse).
    pub value: &'a str,
}

impl<'a> VarDef<'a> {
    pub fn parse(value: &'a str) -> Result<Self, VarDefParseError> {
        def_var_parser().parse(value).into_result().map_err(|err| {
            assert!(err.len() > 0);
            return VarDefParseError::ParseFailed(err[0].to_string());
        })
    }
}

/// Parses a string into a variable definition.
fn def_var_parser<'a>() -> impl Parser<'a, &'a str, VarDef<'a>, SimpleError<'a>> {
    let string = none_of("$=").repeated().at_least(1).to_slice();

    string
        .then_ignore(one_of('='))
        .then(string)
        .map(|(a, b)| VarDef { key: a, value: b })
}

/// Collection of variables defined intrinsicly or via `-Dvar=value`.
#[derive(Clone, Debug, bincode::Encode, bincode::Decode, serde::Serialize, serde::Deserialize)]
pub struct VarDefs {
    /// Named arguments.
    pub defs: HashMap<String, String>,
}

impl VarDefs {
    /// Constructs a new empty set of var definitions.
    pub fn new() -> Self {
        Self {
            defs: HashMap::new(),
        }
    }

    /// Registers a new var definition.
    pub fn insert(&mut self, def: VarDef) {
        self.defs.insert(def.key.to_string(), def.value.to_string());
    }
}

impl Default for VarDefs {
    fn default() -> Self {
        Self::new()
    }
}

/// Error while parsing and transforming strings in a struct.
#[derive(Error, Debug)]
pub enum TransformError {
    #[error("key {0} is not defined")]
    KeyDoesNotExist(String),
    #[error("parsing string failed with: {0}")]
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
    fn transform(&self, vars: &VarDefs) -> Result<Self::Output, TransformError>;
}

impl Transform for str {
    type Output = String;

    fn transform(&self, vars: &VarDefs) -> Result<Self::Output, TransformError> {
        let parser = token_stream_parser();
        let tokens = parser.parse(self).into_result().map_err(|err| {
            assert!(err.len() > 0);
            return TransformError::TokenParseFailed(err[0].to_string());
        })?;

        let mut result = String::new();

        for token in tokens {
            match token {
                Token::String(string) => result.push_str(string),
                Token::Ref(key) => {
                    let Some(name) = vars.defs.get(key) else {
                        return Err(TransformError::KeyDoesNotExist(key.to_string()));
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

    fn transform(&self, vars: &VarDefs) -> Result<Vec<T::Output>, TransformError> {
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
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, bincode::Encode, bincode::Decode)]
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

    fn transform(&self, vars: &VarDefs) -> Result<Self::Output, TransformError> {
        Ok(FloatVar::Inline(match self {
            FloatVar::Inline(v) => *v,
            FloatVar::Script(pos) => pos.transform(vars)?.parse::<f64>()?,
        }))
    }
}

/// A unsigned integer argument that can either be provided via a configuration file
/// or as a config variable.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, bincode::Encode, bincode::Decode)]
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

    fn transform(&self, vars: &VarDefs) -> Result<Self::Output, TransformError> {
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
    /// Reference to variable of the form `${name}`.
    Ref(&'src str),
}

/// Parser of a string into a token.
fn token_parser<'a>() -> impl Parser<'a, &'a str, Token<'a>, SimpleError<'a>> {
    let multidigit_named = just('$')
        .ignore_then(
            none_of('}')
                .repeated()
                .to_slice()
                .delimited_by(just('{'), just('}')),
        )
        .map(|r| Token::Ref(r));

    // let string = none_of('$').repeated().to_slice().map(|r| Token::String(r));

    let misc_string = none_of('$')
        .repeated()
        .at_least(1)
        .to_slice()
        .map(|r| Token::String(r));

    let token = choice((multidigit_named, misc_string));

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
    fn var_def_parsing() {
        assert_eq!(
            VarDef::parse("key=value"),
            Ok(VarDef {
                key: "key",
                value: "value"
            })
        );
        assert_eq!(
            VarDef::parse(" 1@e^ ^9( = DFFS"),
            Ok(VarDef {
                key: " 1@e^ ^9( ",
                value: " DFFS"
            })
        );

        assert!(VarDef::parse("$$=value").is_err());
        assert!(VarDef::parse("=value").is_err());
        assert!(VarDef::parse("=").is_err());
    }

    #[test]
    fn token_parsing() {
        let token = token_parser();

        assert_eq!(token.parse("${100}").into_result(), Ok(Token::Ref("100")));
        assert_eq!(
            token.parse("${1hello}").into_result(),
            Ok(Token::Ref("1hello"))
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
            tokens.parse("hello${1}${name}").into_result(),
            Ok(vec![
                Token::String("hello"),
                Token::Ref("1"),
                Token::Ref("name")
            ])
        );
        assert_eq!(
            tokens.parse("${1}00na${}").into_result(),
            Ok(vec![Token::Ref("1"), Token::String("00na"), Token::Ref("")])
        );
    }

    #[test]
    fn basic_transform() {
        // Set up context
        let mut named = HashMap::new();
        named.insert("arg1".to_string(), "10.0".to_string());
        named.insert("0h@!!".to_string(), "11.0".to_string());
        let ctx = VarDefs { defs: named };

        assert_eq!(
            "some/${arg1}".transform(&ctx).unwrap(),
            "some/10.0".to_string()
        );
        assert_eq!("${0h@!!}".transform(&ctx).unwrap(), "11.0".to_string());
    }
}
