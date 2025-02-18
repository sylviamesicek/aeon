use proc_macro2::TokenStream;
use quote::quote;

use crate::TensorInput;

pub fn tensor(
    TensorInput {
        free,
        contract,
        assign,
    }: TensorInput,
) -> TokenStream {
    let right = assign.right;
    let left = assign.left;

    let mut inner = quote! {
        #left += #right;
    };

    for index in contract.iter() {
        inner = quote! {
            for #index in 0..TENSOR_DIM {
                #inner
            }
        }
    }

    let mut outer = quote! {
        #left = 0.0;
        #inner
    };

    for index in free.iter() {
        outer = quote! {
            for #index in 0..TENSOR_DIM {
                #outer
            }
        }
    }

    outer
}

#[cfg(test)]
mod tests {
    use super::tensor;
    use crate::TensorInput;
    use quote::quote;

    #[test]
    fn tensor_test() {
        let input = quote!(i, j; k => a[i][j] = b[i][j][k] * c[k]).into();
        println!("Input {}\n", input);
        let parsed = syn::parse2::<TensorInput>(input).unwrap();
        let output = tensor(parsed);
        println!("Output {}", output);
    }
}

// pub fn tensor_op(block: &syn::Block) -> syn::Block {
//     let mut stmts = Vec::new();

//     'stmts: for stmt in block.stmts.iter() {
//         'filter: loop {
//             let Stmt::Expr(expr, Some(_semi)) = stmt else {
//                 break 'filter;
//             };
//             // Only check assign expressions
//             let Expr::Assign(assign) = expr else {
//                 break 'filter;
//             };

//             // Check if statement has an attribute matching #[tensor]
//             let Some(_attr) = assign
//                 .attrs
//                 .iter()
//                 .enumerate()
//                 .flat_map(|(i, attr)| {
//                     let is_tensor_attr = attr
//                         .path()
//                         .get_ident()
//                         .map(|ident| ident == "tensor")
//                         .unwrap_or(false);

//                     if is_tensor_attr {
//                         Some((i, attr))
//                     } else {
//                         None
//                     }
//                 })
//                 .next()
//                 .clone()
//             else {
//                 continue;
//             };

//             let left = assign.left.clone();
//             let right = assign.right.clone();

//             // stmts.push(syn::parse_quote! {
//             //     let result = #right;

//             //     // a = b;
//             // });

//             continue 'stmts;
//         }

//         stmts.push(stmt.clone());
//     }

//     syn::Block {
//         brace_token: block.brace_token.clone(),
//         stmts,
//     }
// }

// pub fn tensor(attr: syn::Arm, func: syn::ItemFn) -> TokenStream {
//     let Pat::Lit(dim) = attr.pat.clone() else {
//         return quote_spanned! {
//             attr.span() => compile_error!("#[tensor(N => (i, j, ...))] expects N to be a integer literal.");
//         };
//     };
//     // let Lit::Int(value) = expr.lit else {
//     //     return quote_spanned! {
//     //         attr.span() => compile_error!("#[tensor(N => (i, j, ...))] expects N to be a integer literal.");
//     //     };
//     // };
//     // let Ok(dimension) = value.base10_parse::<usize>() else {
//     //     return quote_spanned! {
//     //         attr.span() => compile_error!("#[tensor(N => (i, j, ...))] expects N to be a integer literal.");
//     //     };
//     // };

//     let Expr::Tuple(tuple) = (&*attr.body).clone() else {
//         return quote_spanned! {
//             attr.span() => compile_error!("#[tensor(N => (i, j, ...))] expects a tuple of indices on the right hand side");
//         };
//     };

//     // Enumerate indices
//     let mut indices = Vec::new();
//     for elem in tuple.elems.iter() {
//         let Expr::Path(expr) = elem else {
//             return quote_spanned! {
//                 elem.span() => compile_error!("#[tensor(N => (i, j, ...))] expects a tuple of indices on the right hand side");
//             };
//         };

//         let Some(ident) = expr.path.get_ident() else {
//             return quote_spanned! {
//                 elem.span() => compile_error!("#[tensor(N => (i, j, ...))] expects a tuple of indices on the right hand side");
//             };
//         };

//         indices.push(ident.clone());
//     }

//     for stmt in func.block.stmts.iter() {
//         let Stmt::Expr(expr, _semi) = stmt else {
//             continue;
//         };
//         let Expr::Assign(assign) = expr else {
//             continue;
//         };

//         // for attrib in assign.attrs.iter().filter_map(|attrib| {
//         //     let path = attrib.path();
//         // }) {
//         //     attrib.
//         // }
//     }

//     func.into_token_stream()
// }
