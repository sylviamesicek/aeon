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

    let mut outer = if contract.len() == 0 {
        quote! {
            #left = #right;
        }
    } else {
        let mut inner = quote! {
            #left += #right;
        };

        for index in contract.iter() {
            inner = quote! {
                for #index in 0..tensor_dim {
                    #inner
                }
            }
        }

        quote! {
            #left = 0.0;
            #inner
        }
    };

    for index in free.iter() {
        outer = quote! {
            for #index in 0..tensor_dim {
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
    fn free() {
        let input = quote!(i, j => a[i][j] = if i == j { 1.0 } else { 0.0 });
        let output = tensor(syn::parse2::<TensorInput>(input).unwrap());

        assert_eq!(
            output.to_string(),
            quote! {
                for j in 0..tensor_dim {
                    for i in 0..tensor_dim {
                        a[i][j] = if i == j { 1.0 } else { 0.0 };
                    }
                }
            }
            .to_string()
        );
    }

    #[test]
    fn contract() {
        let input = quote!(; i, j => a = k[i][i] * m[j][j]);
        let output = tensor(syn::parse2::<TensorInput>(input).unwrap());

        assert_eq!(
            output.to_string(),
            quote! {
                a = 0.0;
                for j in 0..tensor_dim {
                    for i in 0..tensor_dim {
                        a += k[i][i] * m[j][j];
                    }
                }
            }
            .to_string()
        );
    }

    #[test]
    fn free_and_contract() {
        let input = quote!(i, j; k => a[i][j] = b[i][j][k] * c[k]);
        let output = tensor(syn::parse2::<TensorInput>(input).unwrap());
        assert_eq!(
            output.to_string(),
            quote! {
                for j in 0..tensor_dim {
                    for i in 0..tensor_dim {
                        a[i][j] = 0.0;
                        for k in 0..tensor_dim {
                            a[i][j] += b[i][j][k] * c[k];
                        }
                    }
                }
            }
            .to_string()
        );
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
