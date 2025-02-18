mod derive;
mod lagrange;
mod tensor;

use lagrange::Stencil;
use num::rational::Ratio;
use quote::quote;
use syn::{
    parse::{Parse, Parser},
    parse_macro_input,
    punctuated::Punctuated,
    Expr, ExprAssign, Ident, LitInt, Token,
};

#[proc_macro]
pub fn derivative(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let list = Punctuated::<LitInt, Token![,]>::parse_terminated
        .parse(input)
        .unwrap();

    let mut iter = list.iter();
    let left_lit = iter.next().unwrap();
    let right_lit = iter.next().unwrap();
    let point_lit = iter.next().unwrap();

    let left = left_lit.base10_parse::<u64>().unwrap();
    let right = right_lit.base10_parse::<u64>().unwrap();
    let point = point_lit.base10_parse::<i64>().unwrap();

    let stencils = Stencil::vertex(left, right);
    let weights = stencils.derivative_weights(Ratio::from(point));

    let tokens = weights.iter().map(|ratio| {
        let numerator = *ratio.numer() as f64;
        let denomenator = *ratio.denom() as f64;
        let weight = numerator / denomenator;
        quote!(#weight)
    });

    quote! { [ #(#tokens),* ] }.into()
}

#[proc_macro]
pub fn second_derivative(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let list = Punctuated::<LitInt, Token![,]>::parse_terminated
        .parse(input)
        .unwrap();

    let mut iter = list.iter();
    let left_lit = iter.next().unwrap();
    let right_lit = iter.next().unwrap();
    let point_lit = iter.next().unwrap();

    let left = left_lit.base10_parse::<u64>().unwrap();
    let right = right_lit.base10_parse::<u64>().unwrap();
    let point = point_lit.base10_parse::<i64>().unwrap();

    let stencils = Stencil::vertex(left, right);
    let weights = stencils.second_derivative_weights(Ratio::from(point));

    let tokens = weights.iter().map(|ratio| {
        let numerator = *ratio.numer() as f64;
        let denomenator = *ratio.denom() as f64;
        let weight = numerator / denomenator;
        quote!(#weight)
    });

    quote! { [ #(#tokens),* ] }.into()
}

#[proc_macro]
pub fn prolong(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let list = Punctuated::<LitInt, Token![,]>::parse_terminated
        .parse(input)
        .unwrap();

    let mut iter = list.iter();
    let left_lit = iter.next().unwrap();
    let right_lit = iter.next().unwrap();
    let point_lit = iter.next().unwrap();

    let left = left_lit.base10_parse::<u64>().unwrap();
    let right = right_lit.base10_parse::<u64>().unwrap();
    let point = point_lit.base10_parse::<i64>().unwrap();

    let stencils = Stencil::cell(left, right);
    let weights = stencils.value_weights(Ratio::from(point));

    let tokens = weights.iter().map(|ratio| {
        let numerator = *ratio.numer() as f64;
        let denomenator = *ratio.denom() as f64;

        quote!(#numerator / #denomenator)
    });

    quote! { [ #(#tokens),* ] }.into()
}

struct TensorInput {
    free: Punctuated<Ident, Token![,]>,
    contract: Punctuated<Ident, Token![,]>,
    assign: ExprAssign,
}

impl TensorInput {
    fn _parse_dimension(input: syn::parse::ParseStream) -> syn::Result<Expr> {
        if let Ok(expr) = input.parse::<syn::ExprPath>() {
            _ = input.parse::<Token![|]>()?;
            return Ok(Expr::Path(expr));
        }

        let expr = input.parse::<syn::ExprLit>()?;
        _ = input.parse::<Token![|]>()?;

        Ok(Expr::Lit(expr))
    }
}

impl Parse for TensorInput {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        // let dimension = Self::parse_dimension(input)?;

        if input.fork().parse::<Token![;]>().is_ok() {
            let _ = input.parse::<Token![;]>()?;
            let contract = Punctuated::<Ident, Token![,]>::parse_separated_nonempty(input)?;
            // let contract = input.parse_terminated(Ident::parse, Token![,])?;
            let _ = input.parse::<Token![=>]>()?;
            let assign = input.parse::<ExprAssign>()?;
            return Ok(TensorInput {
                free: Punctuated::new(),
                contract,
                assign,
            });
        }
        // Parse free indices
        let free = Punctuated::<Ident, Token![,]>::parse_separated_nonempty(input)?;
        // Fork
        if input.fork().parse::<Token![=>]>().is_ok() {
            let _ = input.parse::<Token![=>]>()?;
            let assign = input.parse::<ExprAssign>()?;
            return Ok(TensorInput {
                free,
                contract: Punctuated::new(),
                assign,
            });
        }
        let _ = input.parse::<Token![;]>()?;
        let contract = Punctuated::<Ident, Token![,]>::parse_separated_nonempty(input)?;
        let _ = input.parse::<Token![=>]>()?;
        let assign = input.parse::<ExprAssign>()?;
        Ok(TensorInput {
            free,
            contract,
            assign,
        })
    }
}

/// A macro for using einstein summation notation.
#[proc_macro]
pub fn tensor(item: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let input = parse_macro_input!(item as TensorInput);
    tensor::tensor(input).into()
}

#[proc_macro_derive(SystemLabel)]
pub fn derive_system_label(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let input = syn::parse_macro_input!(input as syn::DeriveInput);
    derive::system_label_impl(input).into()
}
