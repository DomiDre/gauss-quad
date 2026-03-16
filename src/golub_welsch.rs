//! Contains an implementation of the Golub-Welsch algorithm for the computation of nodes and weights of Gaussian quadrature.

use crate::{Node, Weight};

use alloc::boxed::Box;
use core::num::NonZeroUsize;
use nalgebra::{Dyn, SquareMatrix, VecStorage};

/// Applies the Golub-Welsh algorithm to determine nodes and weights of Gaussian quadrature nodes.
///
/// Given how to construct the diagonal and off-diagonal elements of the companion matrix, this function
/// computes the nodes and weights by solving the eigenvalue problem.
/// See Gil, Segura, Temme - Numerical Methods for Special Functions.
///
/// The functions that compute the elements are given the index of the element along the diagonal.
/// The `diag` function is given the index in the range `0..degree`, while the `off_diag` function
/// if given the index in the range `0..degree - 1`.
///
/// The elements of the first row of the resulting eigenvalue matrix is multiplied by x^2*`scale_factor`.
///
/// The node-weight pairs are sorted by the node.
pub fn golub_welsch<D, O>(
    degree: NonZeroUsize,
    mut diag: D,
    off_diag: O,
    scale_factor: f64,
) -> Box<[(Node, Weight)]>
where
    D: FnMut(usize) -> f64,
    O: Fn(usize) -> f64,
{
    let mut companion_matrix =
        SquareMatrix::<f64, Dyn, VecStorage<f64, Dyn, Dyn>>::zeros(degree.get(), degree.get());

    for idx in 0..degree.get() - 1 {
        companion_matrix[(idx, idx)] = diag(idx);
        let off_diag_elem = off_diag(idx);
        companion_matrix[(idx, idx + 1)] = off_diag_elem;
        companion_matrix[(idx + 1, idx)] = off_diag_elem;
    }
    companion_matrix[(degree.get() - 1, degree.get() - 1)] = diag(degree.get() - 1);

    let eigen = companion_matrix.symmetric_eigen();

    let mut node_weight_pairs: Box<[(Node, Weight)]> = eigen
        .eigenvalues
        .iter()
        .copied()
        .zip(
            eigen
                .eigenvectors
                .row(0)
                .iter()
                .map(|&x| x * x * scale_factor),
        )
        .collect();

    node_weight_pairs.sort_unstable_by(|(node1, _), (node2, _)| node1.total_cmp(node2));

    node_weight_pairs
}
