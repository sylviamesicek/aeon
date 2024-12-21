use proc_macro2::{Ident, Span, TokenStream};
use quote::{quote, quote_spanned};
use syn::{spanned::Spanned, DeriveInput, Fields};

pub fn system_label_impl(input: DeriveInput) -> TokenStream {
    let ident = &input.ident;

    if input.generics.lt_token.is_some() || input.generics.gt_token.is_some() {
        return quote_spanned! {
            input.generics.span() => compile_error!("#[derive(SystemLabel)] does not support generic labels");
        };
    }

    let visibility = &input.vis;

    let mut system_name = ident.to_string();
    system_name.push_str("System");
    let system_ident = Ident::new(&system_name, Span::call_site());

    match &input.data {
        syn::Data::Struct(syn::DataStruct { fields, .. }) => {
            if !matches!(fields, Fields::Unit) {
                return quote_spanned! {
                    ident.span() => compile_error!("#[derive(SystemLabel)] only supported for unit structs.");
                };
            }

            quote! {
                #[derive(Clone, Default)]
                #visibility struct #system_ident;

                #[automatically_derived]
                impl ::aeon::system::System for #system_ident {
                    const NAME: &'static str = stringify!(#ident);

                    type Label = #ident;

                    fn count(&self) -> usize {
                        1
                    }

                    fn enumerate(&self) -> impl Iterator<Item = #ident> {
                        ::std::iter::once(#ident)
                    }

                    fn label_index(&self, _label: #ident) -> usize {
                        0
                    }

                    fn label_name(&self, _label: #ident) -> String {
                        stringify!(#ident).to_string()
                    }

                    fn label_from_index(&self, _index: usize) -> #ident {
                        #ident
                    }
                }
            }
        }
        syn::Data::Enum(syn::DataEnum { variants, .. }) => {
            let variant_count = variants.len();

            let mut fields = quote! {};
            let mut field_indices = quote! {};
            let mut field_names = quote! {};

            for (i, variant) in variants.iter().enumerate() {
                if variant.discriminant.is_some() {
                    return quote_spanned! {
                        variant.span() => compile_error!("#[derive(SystemLabel)] does not support explicit discrimanents.");
                    };
                }
                if !matches!(variant.fields, Fields::Unit) {
                    return quote_spanned! {
                        variant.span() => compile_error!("#[derive(SystemLabel)] only supports unit enums variants.");
                    };
                }

                let var_ident = &variant.ident;

                fields.extend(quote!(#ident::#var_ident,));
                field_indices.extend(quote!(#ident::#var_ident => #i,));
                field_names.extend(quote!(#ident::#var_ident => stringify!(#var_ident),));
            }

            quote! {
                #[derive(Clone, Default)]
                #visibility struct #system_ident;

                #[automatically_derived]
                impl ::aeon::system::System for #system_ident {
                    const NAME: &'static str = stringify!(#ident);

                    type Label = #ident;

                    fn count(&self) -> usize {
                        #variant_count
                    }

                    fn enumerate(&self) -> impl Iterator<Item = #ident> {
                        [#fields].into_iter()
                    }

                    fn label_index(&self, label: #ident) -> usize {
                        match label {
                            #field_indices
                        }
                    }

                    fn label_name(&self, label: #ident) -> String {
                        match label {
                            #field_names
                        }.to_string()
                    }

                    fn label_from_index(&self, index: usize) -> #ident {
                        [#fields][index]
                    }
                }
            }
        }
        syn::Data::Union(_) => quote_spanned! {
            ident.span() => compile_error!("#[derive(SystemLabel)] only supported for structs and enums.");
        },
    }
}
