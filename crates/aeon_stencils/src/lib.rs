mod lagrange;

use lagrange::Stencil;
use num::{rational::Ratio, Zero};

use quote::quote;
use syn::parse::Parser;
use syn::Token;
use syn::{parse_macro_input, punctuated::Punctuated, LitInt};

#[proc_macro]
pub fn centered_derivative(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let extent_literal: LitInt = parse_macro_input!(input as LitInt);
    let extent = extent_literal.base10_parse::<u64>().unwrap();

    let stencils = Stencil::centered(extent);
    let weights = stencils.derivative_weights(Ratio::zero());

    let tokens = weights.iter().map(|ratio| {
        let numerator = *ratio.numer() as f64;
        let denomenator = *ratio.denom() as f64;

        quote!(#numerator / #denomenator)
    });

    quote! { [ #(#tokens),* ] }.into()
}

#[proc_macro]
pub fn centered_second_derivative(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let extent_literal: LitInt = parse_macro_input!(input as LitInt);
    let extent = extent_literal.base10_parse::<u64>().unwrap();

    let stencils = Stencil::centered(extent);
    let weights = stencils.second_derivative_weights(Ratio::zero());

    let tokens = weights.iter().map(|ratio| {
        let numerator = *ratio.numer() as f64;
        let denomenator = *ratio.denom() as f64;

        quote!(#numerator / #denomenator)
    });

    quote! { [ #(#tokens),* ] }.into()
}

#[proc_macro]
pub fn boundary_derivative(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let list = Punctuated::<LitInt, Token![,]>::parse_terminated
        .parse(input)
        .unwrap();

    let mut iter = list.iter();
    let support_lit = iter.next().unwrap();
    let point_lit = iter.next().unwrap();

    let support = support_lit.base10_parse::<u64>().unwrap();
    let point = point_lit.base10_parse::<i64>().unwrap();

    let stencils = Stencil::backward(support);
    let weights = stencils.derivative_weights(Ratio::from(point));

    let tokens = weights.iter().rev().map(|ratio| {
        let numerator = *ratio.numer() as f64;
        let denomenator = *ratio.denom() as f64;

        quote!(#numerator / #denomenator)
    });

    quote! { [ #(#tokens),* ] }.into()
}

#[proc_macro]
pub fn boundary_derivative_neg(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let list = Punctuated::<LitInt, Token![,]>::parse_terminated
        .parse(input)
        .unwrap();

    let mut iter = list.iter();
    let support_lit = iter.next().unwrap();
    let point_lit = iter.next().unwrap();

    let support = support_lit.base10_parse::<u64>().unwrap();
    let point = point_lit.base10_parse::<i64>().unwrap();

    let stencils = Stencil::backward(support);
    let weights = stencils.derivative_weights(Ratio::from(point));

    let tokens = weights.iter().rev().map(|ratio| {
        let numerator = -(*ratio.numer() as f64);
        let denomenator = *ratio.denom() as f64;

        quote!(#numerator / #denomenator)
    });

    quote! { [ #(#tokens),* ] }.into()
}

#[proc_macro]
pub fn boundary_second_derivative(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let list = Punctuated::<LitInt, Token![,]>::parse_terminated
        .parse(input)
        .unwrap();

    let mut iter = list.iter();
    let support_lit = iter.next().unwrap();
    let point_lit = iter.next().unwrap();

    let support = support_lit.base10_parse::<u64>().unwrap();
    let point = point_lit.base10_parse::<i64>().unwrap();

    let stencils = Stencil::backward(support);
    let weights = stencils.second_derivative_weights(Ratio::from(point));

    let tokens = weights.iter().rev().map(|ratio| {
        let numerator = *ratio.numer() as f64;
        let denomenator = *ratio.denom() as f64;

        quote!(#numerator / #denomenator)
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
