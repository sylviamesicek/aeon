use proc_macro2::TokenStream;
use quote::{quote, quote_spanned};
use syn::{spanned::Spanned, DeriveInput, Fields};

pub fn system_label_impl(input: DeriveInput) -> TokenStream {
    let ident = &input.ident;

    if input.generics.lt_token.is_some() || input.generics.gt_token.is_some() {
        return quote_spanned! {
            input.generics.span() => compile_error!("#[derive(SystemLabel)] does not support generic labels");
        };
    }

    match &input.data {
        syn::Data::Struct(syn::DataStruct { fields, .. }) => {
            if !matches!(fields, Fields::Unit) {
                return quote_spanned! {
                    ident.span() => compile_error!("#[derive(SystemLabel)] only supported for unit structs.");
                };
            }

            quote! {
                #[automatically_derived]
                impl ::aeon::system::SystemLabel for #ident {
                    const SYSTEM_NAME: &'static str = stringify!(#ident);

                    fn name(&self) -> String {
                        stringify!(#ident).to_string()
                    }

                    fn index(&self) -> usize {
                        0
                    }

                    type FieldLike<T> = ::aeon::system::SystemArray<T, 1, Self>;

                    fn fields() -> impl Iterator<Item = Self> {
                        [Self].into_iter()
                    }

                    fn field_from_index(_index: usize) -> Self {
                        Self
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

                let ident = &variant.ident;

                fields.extend(quote!(Self::#ident,));
                field_indices.extend(quote!(Self::#ident => #i,));
                field_names.extend(quote!(Self::#ident => stringify!(#ident),));
            }

            quote! {
                #[automatically_derived]
                impl ::aeon::system::SystemLabel for #ident {
                    const SYSTEM_NAME: &'static str = stringify!(#ident);

                    fn name(&self) -> String {
                        match self {
                            #field_names
                        }.to_string()
                    }

                    fn index(&self) -> usize {
                        match self {
                            #field_indices
                        }
                    }

                    type FieldLike<T> = SystemArray<T, #variant_count, Self>;

                    fn fields() -> impl Iterator<Item = Self> {
                        [#fields].into_iter()
                    }

                    fn field_from_index(index: usize) -> Self {
                        [#fields][index].clone()
                    }

                }
            }
        }
        syn::Data::Union(_) => quote_spanned! {
            ident.span() => compile_error!("#[derive(SystemLabel)] only supported for structs and enums.");
        },
    }
}
