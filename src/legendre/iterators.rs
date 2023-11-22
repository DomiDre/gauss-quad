use super::GaussLegendre;
use crate::{impl_data_api, slice_iter_impl};

impl_data_api! {GaussLegendre, GaussLegendreNodes, GaussLegendreWeights, GaussLegendreIter, GaussLegendreIntoIter}
