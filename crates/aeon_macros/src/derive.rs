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
                    const NAME: &'static str = stringify!(#ident);

                    type FieldLike<T> = [T; 1];

                    fn fields() -> ::aeon::array::Array<Self::FieldLike<Self>> {
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
                    const NAME: &'static str = stringify!(#ident);

                    type FieldLike<T> = [T; #variant_count];

                    fn fields() -> ::aeon::array::Array<Self::FieldLike<Self>> {
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
        }
        syn::Data::Union(_) => quote_spanned! {
            ident.span() => compile_error!("#[derive(SystemLabel)] only supported for structs and enums.");
        },
    }
}
