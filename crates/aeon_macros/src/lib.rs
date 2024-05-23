mod lagrange;

use lagrange::Stencil;
use num::rational::Ratio;

use quote::quote;
use syn::parse::Parser;
use syn::Token;
use syn::{punctuated::Punctuated, LitInt};

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

        quote!(#numerator / #denomenator)
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

#[proc_macro_derive(SystemLabel)]
pub fn derive_system_label(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let input = syn::parse_macro_input!(input as syn::DeriveInput);
    let ident = &input.ident;

    match &input.data {
        syn::Data::Struct(syn::DataStruct { fields, .. }) => {
            assert!(
                fields.len() == 0,
                "#[derive(SystemLabel)] only implemented for structs with no fields."
            );

            quote! {
                #[automatically_derived]
                impl SystemLabel for #ident {
                    const NAME: &'static str = stringify!(#ident);

                    type FieldLike<T> = [T; 1];

                    fn fields() -> Array<Self::FieldLike<Self>> {
                        [Self].into()
                    }

                    fn field_index(&self) -> usize {
                        0
                    }

                    fn field_name(&self) -> String {
                        stringify!(#ident).to_string()
                    }
                }
            }
            .into()
        }
        syn::Data::Enum(syn::DataEnum { variants, .. }) => {
            let variant_count = variants.len();

            let mut fields = quote! {};
            let mut field_indices = quote! {};
            let mut field_names = quote! {};

            for (i, variant) in variants.iter().enumerate() {
                assert!(
                    variant.discriminant.is_none(),
                    "#[derive(SystemLabel)] does not support explicit discrimanents."
                );
                assert!(
                    matches!(variant.fields, syn::Fields::Unit),
                    "#[derive(SystemLabel)] only supports unit enum variants."
                );

                let ident = &variant.ident;

                fields.extend(quote!(Self::#ident,));
                field_indices.extend(quote!(Self::#ident => #i,));
                field_names.extend(quote!(Self::#ident => stringify!(#ident)));
            }

            quote! {
                #[automatically_derived]
                impl SystemLabel for #ident {
                    const NAME: &'static str = stringify!(#ident);

                    type FieldLike<T> = [T; #variant_count];

                    fn fields() -> Array<Self::FieldLike<Self>> {
                        [#fields].into()
                    }

                    fn field_index(&self) -> usize {
                        match self {
                            #field_indices
                        }
                    }

                    fn field_name(&self) -> String {
                        match self {
                            #field_names
                        }.to_string()
                    }
                }
            }
            .into()
        }
        syn::Data::Union(_) => {
            panic!("#[derive(SystemLabel)] only implemented for structs and enums.")
        }
    }
}
